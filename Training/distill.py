from transformers import GPTNeoModel, AutoModelForCausalLM,\
    GPT2Tokenizer, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader 
from lm_dataformat import *
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize, cross_entropy
from torch.nn import DataParallel
from auto_tqdm import tqdm
from get_args import get_args
import deepspeed

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create model, set neo_hidden
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
neo_hidden = model.config.hidden_size
#resize token embeddings. Two extra tokens
model.resize_token_embeddings(len(tokenizer)+2)

#Set up deep speed
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=model,
                                                        model_parameters=model.parameters())
#Cast to device
model_engine.to(device)

#Initialize a random projection matrix
clip_hidden = 512
projection = torch.nn.Linear(neo_hidden, clip_hidden, bias=False).to(device)


#hparams
temperature = 1.0
learning_rate = 5e-5
weight_decay = 0
grad_accum = 2
clip_bs = 8
lambda_coeff = 1.0 #relative scale for contrastive loss

temp_tensor = torch.tensor(temperature).to(device)

#pytorch dataset for clip juicing
class DistillDataset(IterableDataset):
    def __init__(self,\
        tokenizer, clip_batch_size,
        clip_dataset_dir, pile_dataset_dir,
        special_token = "<|CLIP|>", steps = 1e6):
        self.clip_dataset_dir = clip_dataset_dir
        self.pile_dataset_dir = pile_dataset_dir

        self.clip_rdr = Reader(self.clip_dataset_dir).stream_data(get_meta=True)
        self.pile_rdr = Reader(self.pile_dataset_dir).stream_data(get_meta=True)

        #Steps is the total number of elements we should use. Half from CLIP, half from AR
        self.steps = steps
        #How many elements are in a single contrastive clip batch
        self.clip_batch_size = clip_batch_size

        #Start on an example of WIT.
        self.cur_clip = True

        #Store special token, add to tokenizer. Remember to resize token embeddings on model!
        self.tokenizer = tokenizer
        self.special_token=special_token
        #Get the index for the special token so that we can adjust the decode mask accordingly.
        self.special_token_idx=len(self.tokenizer)
        self.tokenizer.add_tokens([special_token])
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.i=0
    def __len__(self):
        return int(self.steps)
    def __iter__(self):
        return self
    def __next__(self):
        self.i+=1
        print("DATASET STEP", self.i)
        tok = self.tokenizer
        txts = list()
        img_latents = list()
    
        #Return an element from the pile
        if not self.cur_clip:
            for _ in range(1):

                text, _ =next(self.pile_rdr)
                txts.append(text)
                #Place holder
                img_latents.append([[0]*clip_hidden])
        #Return an element from CLIP
        else:
            txts = list()
            for _ in range(self.clip_batch_size):
                text, img_latent=next(self.clip_rdr)
                #Append special token
                text += "<|CLIP|>"
                txts.append(text)
                img_latents.append([img_latent])

        #Tokenize text
        toks = tok.batch_encode_plus(txts, max_length=1024, truncation=True, return_tensors="pt", pad_to_multiple_of=8, padding=True).to(device)
        #Get the index of the clip tokens.
        clip_idx = (torch.sum(toks.attention_mask, dim=-1).to("cpu") - torch.tensor([1] * len(txts)))
        #Get latent vectors
        latents = torch.cat([torch.tensor(x) for x in img_latents], dim=0).to(device)

        cc = self.cur_clip
        #Flip cur clip
        self.cur_clip = not self.cur_clip 
        return {
            **toks,
            'latent_vecs' : latents,
            'clip_idx' : clip_idx, 
            'use_distill' : cc,
        }
dev='cuda'
#Contrastive loss helper function
def clip_loss(a, b, temp):
    # a ~ (b x d)
    # b ~ (b x d)
#    return torch.tensor(0.).to(dev).requires_grad_(True)
    batch_size, dimension = a.shape

    a_normd = normalize(a, p=2, dim=1).squeeze()
    b_normd = normalize(b, p=2, dim=1).squeeze()
    logits = torch.einsum('i d, j d -> i j', a_normd, b_normd) * temp.exp()

    labels = torch.arange(batch_size).to(device)

    loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
    
    return loss / 2.0

def ar_loss(out_embeds, inp):
    # inp :: [b, seq]
#    return torch.tensor(0.).to(dev).requires_grad_(True)
    logprobs = F.log_softmax(out_embeds['logits'].squeeze(0), dim=-1)
    # logprobs :: [b, seq, vocab]

    pred = logprobs[:, :-1]
    tgt = inp.squeeze(0)[:, 1:]

    is_clip_or_padding_token = tgt >= 50257
    
    print(pred.shape, tgt.shape)
    return F.nll_loss(pred.squeeze(0), tgt.squeeze(0))

    logits = torch.gather(pred, 2, tgt.unsqueeze(-1)).squeeze(-1) # [batch, seq-1]

    # remove loss of clip-token
    logits *= 1 - is_clip_or_padding_token.to(torch.int)

    return -logits.sum()

#Load dataset
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../../clip_latents_100k.jsonl.zst",\
    pile_dataset_dir = "../../../val.jsonl.zst")
loader = DataLoader(dataset=data, batch_size=1)

#Check to see if a checkpoint exists. if it does, load that. Otherwise assume we are on step zero.
try:
    _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
    step = client_sd['step']
    loader_to_step(loader, step+1)
except:
    step = 0
#Set up progress bar
pbar = tqdm(enumerate(loader), total=len(data))
loss_progress = 0.0

#Update the pbar description every 20 batches
report_loss_every = 20
#save every 10000 batches
save_every = 10000

#input("waiting on input")
import gc
def train_step(batch, data_elem):
    if batch < 50: 
        print('batch skip',batch)
        return
    global loss_progress
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=device, abbreviated=False))
#    time.sleep(3)
    print(data_elem['input_ids'].shape)
    print(data_elem)
#   if data_elem['input_ids'].shape[-1] > 1000: return
    model_input = {
        'input_ids':data_elem['input_ids'],
        'attention_mask':data_elem['attention_mask'],
    }
    loss = None
    # compute model once for both CLIP and AR
#    return 

    model_out = model_engine(**model_input, return_dict=True, output_hidden_states=True)
    # denug shapes
    #print([(k, v.shape if isinstance(v, torch.Tensor) else v) for k, v in data_elem.items()])

    #If we are currently using contrastive loss
    if data_elem['use_distill']:
        #out_embeds ~ (b x seq_len x hidden_size)
        idx = data_elem['clip_idx']
        last_layer = model_out["hidden_states"][-1].squeeze() # -1 for last layer
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(device)
        for i,j in enumerate(idx.tolist()[0]):
            clip_embeds[i] = last_layer[i][j]

        #Project to the correct size
        clip_embeds = projection(clip_embeds)
        #Compute contrastive loss
        loss = lambda_coeff * clip_loss(clip_embeds,  data_elem['latent_vecs'], temp_tensor)
        loss_progress += float(loss.detach().cpu().item())

        del idx, last_layer, clip_embeds
    else:
        #compute AR loss if Pile data
        n_text_toks = data_elem['clip_idx'].sum().detach()
        loss = ar_loss(model_out, model_input['input_ids']) / n_text_toks

    torch.cuda.empty_cache()
    model_engine.backward(loss)

        #loss = model_engine(batch)
    #model_engine.backward(loss)
    #loss_progress += loss.detach().cpu().item()
    #del loss
    #Accumulate gradients
    if (batch+1)%grad_accum==0:
        model_engine.step()
        model_engine.zero_grad()

    #Update loss progress
    if (batch+1)%report_loss_every==0:
        loss_progress /= float(report_loss_every)
        pbar.set_description("Current loss: " + str(loss_progress))
        loss_progress = 0.0
    #Save model
    if (batch+1)%save_every==0:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
        #model.save_pretrained("GPT-Neo-Enriched"+str(batch+1))
        tokenizer.save_pretrained("GPT-Neo-Enriched"+str(batch+1))
    del loss, model_input, data_elem, model_out
    print(torch.cuda.memory_allocated(0))
    torch.cuda.empty_cache()

for batch, data_elem in pbar:
    train_step(batch, data_elem)
model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
#model.save_pretrained("GPT-Neo-Enriched")
tokenizer.save_pretrained("GPT-Neo-Enriched")
