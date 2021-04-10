from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2TokenizerFast, GPTNeoConfig, AdamW
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
import os
from random import randint
import math


# set random
import torch
torch.manual_seed(52)
import random
random.seed(52)
import numpy as np
np.random.seed(52)


#enable tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#create model, set neo_hidden
conf = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
conf.gradient_checkpointing = True
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", config=conf)
model.training = True

tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B")
neo_hidden = model.config.hidden_size
#resize token embeddings. Two extra tokens
model.resize_token_embeddings(len(tokenizer)+2)
#Initialize a random projection matrix
clip_hidden = 512
#Set up deep speed

#Set up our model wrapper, makes saving easier
class projection_model(torch.nn.Module):
    def __init__(self):
        super(projection_model, self).__init__()
        self.fc1 = torch.nn.Linear(neo_hidden, neo_hidden//2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(neo_hidden//2, clip_hidden)
    def forward(self, input_tensor):
        out = self.act(self.fc1(input_tensor))
        return self.fc2(out)

projection=projection_model()
projection.half()

class ModelWrapper(torch.nn.Module):
    def __init__(self, lm, proj):
        super(ModelWrapper, self).__init__()
        self.lm = lm
        self.proj = proj
    def forward(self, **kwargs):
        return self.lm(**kwargs)
wrapper = ModelWrapper(model, projection)
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=wrapper,
                                                        model_parameters=wrapper.parameters())
model_engine.to(model_engine.local_rank)
#projection = projection.to(model_engine.local_rank)



#Set up wandb
if model_engine.local_rank == 0:
    wandb.init(project='speedrun', entity='eleutherai')



#hparams
temperature = 1.0
learning_rate = 5e-5
weight_decay = 0
grad_accum = 2
clip_bs = 20
pile_bs = 1
lambda_coeff = 0.1 #relative scale for contrastive loss

mixing_ratio = 3 #Ratio of pile examples to CLIP examples 

# lambda scheduling
lschedule = "truncated_sine" # truncated_sine, shifted_sine, constant
lperiod = 1000

temp_tensor = torch.tensor(temperature).to(model_engine.local_rank)


if model_engine.local_rank == 0:
    config=wandb.config
    config.learning_rate=learning_rate
    config.temp=temperature
    config.weight_decay=weight_decay
    config.clip_batch_suze=clip_bs
    config.lambda_c=lambda_coeff

    wandb.watch(model)
    try:
        os.makedirs(f"models/{wandb.run.name}")    
        torch.save(wrapper.proj, f"models/{wandb.run.name}/projection.pt")
    except:
        pass


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
        self.mix_step = 0 

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

        use_clip = False
        #Mixing
        self.mix_step += 1
        if self.mix_step % mixing_ratio==0:
            use_clip=True
            self.mix_step=0
        #Return an element from the pile
        if not use_clip:
            for _ in range(pile_bs):
                text, _ =next(self.pile_rdr)
        
                #text split. Check if we are over length. Truncate accordingly
                ts = text.split()
                if len(ts) > 1024:
                    start = randint(0, len(ts)-1024)

                    text = " ".join(ts[start:])
                txts.append(text)

            #Tokenize text
            toks = tok.batch_encode_plus(txts, max_length=1024, truncation=True,\
                padding="max_length", return_tensors="pt").to(model_engine.local_rank)
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
            toks = tok.batch_encode_plus(txts, max_length=128, truncation=True,\
                padding="max_length", return_tensors="pt").to(model_engine.local_rank)
        #Get the index of the clip tokens.
        clip_idx = (torch.sum(toks.attention_mask, dim=-1).to("cpu") - torch.tensor([1] * len(txts)))
        #Get latent vectors
        latents = torch.cat([torch.tensor(x) for x in img_latents], dim=0).to(model_engine.local_rank)
        
        return {
            **toks,
            'latent_vecs' : latents,
            'clip_idx' : clip_idx, 
            'use_distill' : use_clip,
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
    raw_logits = out_embeds['logits'].squeeze(0)
    logprobs = F.log_softmax(
        torch.cat([
            raw_logits[:, :, :50257],
            -1e10 * torch.ones(raw_logits.shape[0], raw_logits.shape[1], 2).to(model_engine.local_rank)
        ], dim=-1)
        , dim=-1).to(torch.float32)
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
    pile_dataset_dir = "../../pile/")
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

#Update the pbar description every 10 batches
report_loss_every = 10
#save every 500 batches
save_every = 500

#generate every
generate_every = 300
generate_template = tokenizer("EleutherAI is", return_tensors="pt").to(model_engine.local_rank)
clip_template = tokenizer("A drawing of a goose holding a knife<|CLIP|>", return_tensors="pt").to(model_engine.local_rank)
clip_idx_template = torch.sum(clip_template['attention_mask'], dim=-1).to("cpu").squeeze().item() - 1
text_so_far = list()
#Generate text samples occasionally
def generate_from_template():
    if model_engine.local_rank == 0:
        out_embeds = model_engine(**clip_template, return_dict=True, output_hidden_states=True)['hidden_states'][-4].squeeze()
        CLIP_state = wrapper.proj(out_embeds[clip_idx_template]).cpu().detach().numpy()
        np.savetxt('clip.out', CLIP_state, delimiter=",")

        #text_so_far.append(["EleutherAI is ", text])
        #wandb.log({"Generated": text})

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

    #If we are currently using contrastive loss
    if data_elem['use_distill']:
        #out_embeds ~ (b x seq_len x hidden_size)
        idx = data_elem['clip_idx']
        last_layer = out_embeds[-4].squeeze() # -1 for second to last layer
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(model_engine.local_rank)
        for i,j in enumerate(idx.tolist()[0]):
            clip_embeds[i] = last_layer[i][j]

        #Project to the correct size
        clip_embeds = wrapper.proj(clip_embeds.to(torch.float16))
        #Compute contrastive loss
        if lschedule == "constant":
            actual_lambda = lambda_coeff
        elif lschedule == "truncated_sine":
            actual_lambda = lambda_coeff * max(0, math.sin(2*math.pi / lperiod * batch))
        elif lschedule == "shifted_sine":
            actual_lambda = lambda_coeff * math.sin(math.pi / lperiod * batch)**2
        else:
            raise NotImplementedError()    
        loss = actual_lambda * clip_loss(clip_embeds,  data_elem['latent_vecs'], temp_tensor)
    else:
        #compute AR loss if Pile data
        n_text_toks = data_elem['clip_idx'].sum()
        loss = ar_loss(model_out, data_elem['input_ids']) / n_text_toks
    #Accumulate loss, check if NaN. If not, update progress
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
    if batch%generate_every==0:
        generate_from_template()
    
    #Update loss progress. For WandB
    if (batch+1)%report_loss_every==0:
        loss_progress /= float(loss_step_count)
        if model_engine.local_rank == 0:
            clip_loss_progress /= float(max(clip_step_count,1))
            clip_step_count = 0

            ar_loss_progress /= float(max(ar_step_count,1))
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
        ckpt_id = (batch+1)
        #model_engine.save_checkpoint(f"models/{wandb.run.name}{ckpt_id}")
        #model_engine.module.save_pretrained(wandb.run.dir)
        # local or global?
        if model_engine.local_rank == 0:
            model_engine.module.lm.save_pretrained(f"models/{wandb.run.name}/{ckpt_id}", ckpt_id)
            torch.save(wrapper.proj, f"models/{wandb.run.name}/projection.pt")
        #model.save_pretrained("GPT-Neo-Enriched"+str(batch+1))
        #tokenizer.save_pretrained("GPT-Neo-Enriched"+str(batch+1))

model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
#model.save_pretrained("GPT-Neo-Enriched")
tokenizer.save_pretrained("GPT-Neo-Enriched")
