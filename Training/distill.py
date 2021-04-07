from transformers import GPTNeoModel, GPTNeoForCausalLM,\
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
import wandb


#enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#create model, set neo_hidden
conf = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
conf.gradient_checkpointing = True
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", config=conf)
model.training = True

tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
neo_hidden = model.config.hidden_size
#resize token embeddings. Two extra tokens
model.resize_token_embeddings(len(tokenizer)+2)
#Set up deep speed
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=model,
                                                        model_parameters=model.parameters())
model_engine.to(model_engine.local_rank)

#Set up wandb
if model_engine.local_rank == 0:
    wandb.init(project='speedrun', entity='eleutherai')

#Initialize a random projection matrix
clip_hidden = 512
projection = torch.nn.Linear(neo_hidden, clip_hidden, bias=False).to(model_engine.local_rank)


#hparams
temperature = 1.0
learning_rate = 5e-5
weight_decay = 0
grad_accum = 2
clip_bs = 48
lambda_coeff = 0.75 #relative scale for contrastive loss

temp_tensor = torch.tensor(temperature).to(model_engine.local_rank)


if model_engine.local_rank == 0:
    config=wandb.config
    config.learning_rate=learning_rate
    config.temp=temperature
    config.weight_decay=weight_decay
    config.clip_batch_suze=clip_bs
    config.lambda_c=lambda_coeff

    wandb.watch(model)


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
    def __len__(self):
        return int(self.steps)
    def __iter__(self):
        return self
    def __next__(self):
        tok = self.tokenizer
        txts = list()
        img_latents = list()
    
        #Return an element from the pile
        if not self.cur_clip:
            text, _ =next(self.pile_rdr)
            txts.append(text)
            #Place holder
            #Tokenize text
            toks = tok.batch_encode_plus(txts, max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(model_engine.local_rank)
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
            toks = tok.batch_encode_plus(txts, max_length=128, truncation=True, padding="max_length", return_tensors="pt").to(model_engine.local_rank)

        #Get the index of the clip tokens.
        clip_idx = (torch.sum(toks.attention_mask, dim=-1).to("cpu") - torch.tensor([1] * len(txts)))
        #Get latent vectors
        latents = torch.cat([torch.tensor(x) for x in img_latents], dim=0).to(model_engine.local_rank)

        cc = self.cur_clip
        #Flip cur clip
        self.cur_clip = not self.cur_clip 
        return {
            **toks,
            'latent_vecs' : latents,
            'clip_idx' : clip_idx, 
            'use_distill' : cc,
        }

#Contrastive loss helper function
def clip_loss(a, b, temp):
    # a ~ (b x d)
    # b ~ (b x d)
    batch_size, dimension = a.shape

    a_normd = normalize(a, p=2, dim=1).squeeze().to(torch.float32)
    b_normd = normalize(b, p=2, dim=1).squeeze().to(torch.float32)
    logits = torch.einsum('i d, j d -> i j', a_normd, b_normd) * temp.exp()

    labels = torch.arange(batch_size).to(model_engine.local_rank)

    loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
    
    return loss / 2.0

def ar_loss(out_embeds, inp):
    # inp :: [b, seq]
    logprobs = F.log_softmax(out_embeds['logits'].squeeze(0), dim=-1).to(torch.float32)
    # logprobs :: [b, seq, vocab]

    pred = logprobs[:, :-1]
    tgt = inp.squeeze(0)[:, 1:]

    is_clip_or_padding_token = tgt >= 50257
    
    logits = torch.gather(pred, 2, tgt.unsqueeze(-1)).squeeze(-1) # [batch, seq-1]

    # remove loss of clip-token
    logits *= 1 - is_clip_or_padding_token.to(torch.int)

    return -logits.sum()

#Load dataset
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../clip/",\
    pile_dataset_dir = "../../val.jsonl.zst")
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
loss_step_count = 0

ar_loss_progress = 0.0
ar_step_count = 0

clip_loss_progress = 0.0
clip_step_count = 0

#Update the pbar description every 20 batches
report_loss_every = 20
#save every 10000 batches
save_every = 3000

for batch, data_elem in pbar:
    torch.cuda.empty_cache()
    model_input = {
        'input_ids':data_elem['input_ids'],
        'attention_mask':data_elem['attention_mask'],
    }
    loss = None
    
    # compute model once for both CLIP and AR
    model_out = model_engine(**model_input, return_dict=True, output_hidden_states=True)
    out_embeds = model_out['hidden_states']
    #print("Layers:\n\n")
    #print(out_embeds[-1])
    #print("Logits:\n\n")
    #print(model_out['logits'])
    # debug shapes
    #print([(k, v.shape if isinstance(v, torch.Tensor) else v) for k, v in data_elem.items()])

    #If we are currently using contrastive loss
    if data_elem['use_distill']:
        #out_embeds ~ (b x seq_len x hidden_size)
        idx = data_elem['clip_idx']
        last_layer = out_embeds[-1].squeeze() # -1 for last layer
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(model_engine.local_rank)
        for i,j in enumerate(idx.tolist()[0]):
            clip_embeds[i] = last_layer[i][j]

        #Project to the correct size
        clip_embeds = projection(clip_embeds)
        #Compute contrastive loss
        loss = lambda_coeff * clip_loss(clip_embeds,  data_elem['latent_vecs'], temp_tensor)
    else:
        #compute AR loss if Pile data
        n_text_toks = data_elem['clip_idx'].sum()
        loss = ar_loss(model_out, data_elem['input_ids']) / n_text_toks
    #loss = model_engine(batch)
    if not torch.any(loss.isnan()):
        model_engine.backward(loss.to(torch.float32))
        model_engine.step()
        loss_progress += loss.to(torch.float32).detach().cpu().item()
        loss_step_count += 1

        if data_elem['use_distill']:
            clip_loss_progress += loss.to(torch.float32).detach().cpu().item()
            clip_step_count += 1
        else:
            ar_loss_progress += loss.to(torch.float32).detach().cpu().item()
            ar_step_count += 1

    
    #Update loss progress
    if (batch+1)%report_loss_every==0:
        loss_progress /= float(loss_step_count)
        if model_engine.local_rank == 0:
            clip_loss_progress /= float(clip_step_count)
            clip_step_count = 0

            ar_loss_progress /= float(ar_step_count)
            ar_step_count = 0
            wandb.log({'CLIP loss': clip_loss_progress, 'AR loss': ar_loss_progress, 'Loss' : loss_progress})
        pbar.set_description("Current loss: " + str(loss_progress))
        loss_progress = 0.0
        clip_loss_progress = 0.0
        ar_loss_progress = 0.0
        loss_step_count = 0 
    #Save model
    if (batch+1)%save_every==0:
        torch.distributed.barrier()
        ckpt_id = loss.item()
        model_engine.save_checkpoint("models/", ckpt_id)
        #model.save_pretrained("GPT-Neo-Enriched"+str(batch+1))
        #tokenizer.save_pretrained("GPT-Neo-Enriched"+str(batch+1))

model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
#model.save_pretrained("GPT-Neo-Enriched")
tokenizer.save_pretrained("GPT-Neo-Enriched")
