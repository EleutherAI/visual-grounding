from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2Tokenizer, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader 
from lm_dataformat import *
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize, cross_entropy
from torch.nn import DataParallel
from auto_tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initalize a smol boi
config = GPTNeoConfig(hidden_size = 128, num_layers = 24, attention_layers = 24)
#create model
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
if torch.cuda.device_count() > 1:
    model_dp = DataParallel(model)
model_dp.to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

#Initialize a random projection matrix
neo_hidden = model.config.hidden_size
clip_hidden = 512
projection = torch.nn.Linear(neo_hidden, clip_hidden, bias=False).to(device)


#hparams
temperature = 1.0
learning_rate = 5e-5
weight_decay = 0
grad_accum = 2
clip_bs = 128
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
        toks = tok.batch_encode_plus(txts, max_length=2048, truncation=True, padding=True, return_tensors="pt").to(device)
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

#Contrastive loss helper function
def clip_loss(a, b, temp):
    # a ~ (b x d)
    # b ~ (b x d)
    batch_size, dimension = a.shape

    a_normd = normalize(a, p=2, dim=1).squeeze()
    b_normd = normalize(b, p=2, dim=1).squeeze()
    logits = torch.einsum('i d, j d -> i j', a_normd, b_normd) * temp.exp()

    labels = torch.arange(batch_size).to(device)

    loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
    
    return loss / 2.0

def ar_loss(model, inp, attn_mask):
    # inp :: [b, seq]
    logprobs = F.log_softmax(model(inp, attention_mask=attn_mask, return_dict=True)['logits'].squeeze(0), dim=-1)
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
    clip_dataset_dir = "../clip_latents_100k.jsonl.zst",\
    pile_dataset_dir = "../val.jsonl.zst")
loader = DataLoader(dataset=data, batch_size=1)

#resize token embeddings
model.resize_token_embeddings(len(data.tokenizer))

#Set up optimizer
opt = AdamW(list(model_dp.parameters()) + list(projection.parameters()), lr=learning_rate, weight_decay=weight_decay)


#Set up progress bar
pbar = tqdm(enumerate(loader), total=len(data))
loss_progress = 0.0

#Update the pbar description every 20 batches
report_loss_every = 20
#save every 10000 batches
save_every = 10000

for batch, data_elem in pbar:
    model_input = {
        'input_ids':data_elem['input_ids'],
        'attention_mask':data_elem['attention_mask'],
    }
    loss = None
    #Used for CLIP. TODO: Fetch AR loss (Leo pls do this)
    out_embeds = model_dp(**model_input, return_dict=True, output_hidden_states=True)['hidden_states']
    
    # debug shapes
    #print([(k, v.shape if isinstance(v, torch.Tensor) else v) for k, v in data_elem.items()])

    #If we are currently using contrastive loss
    if data_elem['use_distill']:
        #out_embeds ~ (b x seq_len x hidden_size)
        idx = data_elem['clip_idx']
        last_layer = out_embeds[-1].squeeze() # -1 for last layer
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(device)
        for i,j in enumerate(idx.tolist()[0]):
            clip_embeds[i] = last_layer[i][j]

        #Project to the correct size
        clip_embeds = projection(clip_embeds)
        #Compute contrastive loss
        loss = lambda_coeff * clip_loss(clip_embeds,  data_elem['latent_vecs'], temp_tensor)

    #compute AR loss
    n_text_toks = data_elem['clip_idx'].sum()
    if loss is not None:
        loss += ar_loss(model_dp, data_elem['input_ids'], data_elem['attention_mask']) / n_text_toks
    else:
        loss = ar_loss(model_dp, data_elem['input_ids'], data_elem['attention_mask']) / n_text_toks

    loss.backward()
    loss_progress += loss.detach().cpu().item()
    
    #Accumulate gradients
    if (batch+1)%grad_accum==0:
        opt.step()
        opt.zero_grad()

    #Update loss progress
    if (batch+1)%report_loss_every==0:
        loss_progress /= float(report_loss_every)
        pbar.set_description("Current loss: " + str(loss_progress))
        loss_progress = 0.0
    #Save model
    if (batch+1)%save_every==0:
        model.save_pretrained("GPT-Neo-Enriched"+str(batch+1))
        tokenizer.save_pretrained("GPT-Neo-Enriched"+str(batch+1))

model.save_pretrained("GPT-Neo-Enriched")
tokenizer.save_pretrained("GPT-Neo-Enriched")
