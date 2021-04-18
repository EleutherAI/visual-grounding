from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2TokenizerFast, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader 
from lm_dataformat import *
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize, cross_entropy
from auto_tqdm import tqdm
from get_args import get_args
import deepspeed
import wandb
import os
from random import randint
import math

#Helper files
from utility import *
from loss import *
from debug_utils import DebugActivationOverflow, DebugOption


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



projection=projection_model(neo_hidden)
projection.half()



wrapper = ModelWrapper(model, projection)
model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                                        model=wrapper,
                                                        model_parameters=wrapper.parameters())
model_engine.to(model_engine.local_rank)
if args.debug:
    debug_overflow = DebugActivationOverflow(model_engine) # original uses self.model so it might need some looking into
temp_tensor = torch.tensor(temperature).to(model_engine.local_rank)


#Load dataset
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../clip/",\
    pile_dataset_dir = "../../pile/", local_rank = model_engine.local_rank)

step=0

#Set up progress bar
pbar = tqdm(enumerate(data), total=len(data))
loss_progress = 0.0
loss_step_count = 0

ar_loss_progress = 0.0
ar_step_count = 0

clip_loss_progress = 0.0
clip_step_count = 0

#reset counters every
reset_every = 25
#save every 20 steps
save_every = 500
#What are the number of optimizer steps we've done
true_step = 0

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
        continue


    #compute AR loss if Pile data
    n_text_toks = data_elem['clip_idx'].sum()
    ar_l =ar_loss(model_out, data_elem['input_ids'], local_rank=model_engine.local_rank) / n_text_toks
    if loss is None:
        loss = ar_l
    else:
        loss += ar_l


    #Accumulate loss, check if NaN. If not, update progress
    if not torch.any(loss.isnan()):
        model_engine.backward(loss.to(torch.float32))
        model_engine.step()
        #Clamp the temperature
        with torch.no_grad():
            wrapper.temperature.clamp_(1./100., 100.)

model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
tokenizer.save_pretrained("GPT-Neo-Enriched")
