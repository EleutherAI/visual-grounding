from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2Tokenizer, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader 
from lm_dataformat import *
import torch
from torch.nn.functional import normalize, cross_entropy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initalize a smol boi
config = GPTNeoConfig(hidden_size = 128, num_layers = 24, attention_layers = 24)
#create model
model = GPTNeoForCausalLM(config).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

#Initialize a random projection matrix
neo_hidden = model.config.hidden_size
clip_hidden = 512
projection = torch.nn.Linear(neo_hidden, clip_hidden).to(device)
for param in projection.parameters():
    param.requires_grad = False


#hparams
temperature = 1.0
learning_rate = 5e-5
weight_decay = 0
grad_accum = 2

temp_tensor = torch.tensor(temperature).to(device)

#pytorch dataset for clip juicing
class DistillDataset(IterableDataset):
    def __init__(self,\
        tokenizer, clip_batch_size,
        clip_dataset_dir, pile_dataset_dir,
        special_token = "<CLIP>", steps = 1e6):
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
        return self.steps
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
                text += "<CLIP>"
                txts.append(text)
                img_latents.append([img_latent])

        #Tokenize text
        toks = tok.batch_encode_plus(txts, max_length=2048, truncation=True, padding=True, return_tensors="pt").to(device)
        #Get the index of the clip tokens.
        clip_idx = torch.sum(toks.attention_mask, dim=-1) - torch.tensor([1] * len(txts)).to(device)
        #Get latent vectors
        latents = torch.stack([torch.tensor(x) for x in img_latents]).to(device)

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

#Load dataset
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = 8,\
    clip_dataset_dir = "../clip_latents_100k.jsonl.zst",\
    pile_dataset_dir = "../val.jsonl.zst")
#resize token embeddings
model.resize_token_embeddings(len(data.tokenizer))

#Set up optimizer
opt = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for batch, data_elem in enumerate(data):
    model_input = {
        'input_ids':data_elem['input_ids'],
        'attention_mask':data_elem['attention_mask'],
    }
    #Used for CLIP. TODO: Fetch AR loss (Leo pls do this)
    out_embeds = model(**model_input, return_dict=True, output_hidden_states=True)['hidden_states']
    #If we are currently using contrastive loss
    if data_elem['use_distill']:
        #out_embeds ~ (b x seq_len x hidden_size)
        idx = data_elem['clip_idx']
        last_layer = out_embeds[0]
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(device)
        for i,j in enumerate(idx):
            clip_embeds[i] = last_layer[i][j]
        #Project to the correct size
        clip_embeds = projection(clip_embeds)
        #Compute loss
        loss = clip_loss(clip_embeds,  data_elem['latent_vecs'], temp_tensor)
        #loss += ar_loss_with_clip_masked #Leo pls
    else:
        pass
        #Do AR loss only
    loss.backward()
    if (batch+1)%grad_accum==0:
        opt.step()
        opt.zero_grad()
    #Validation loop?

    break