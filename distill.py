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
temp_tensor = torch.tensor(temperature).to(model_engine.local_rank)



#Set up wandb
if model_engine.local_rank == 0:
    wandb.init(project='speedrun', entity='eleutherai')

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


#Load dataset
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../clip/",\
    pile_dataset_dir = "../../pile/", local_rank = model_engine.local_rank)

#Check to see if a checkpoint exists. if it does, load that. Otherwise assume we are on step zero.
try:
    _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
    step = client_sd['step']
    loader_to_step(loader, step+1)
except:
    step = 0

#Set up progress bar
pbar = tqdm(enumerate(data), total=len(data))
loss_progress = 0.0
loss_step_count = 0

ar_loss_progress = 0.0
ar_step_count = 0

clip_loss_progress = 0.0
clip_step_count = 0

#save every 20 steps
save_every = 20
#What are the number of optimizer steps we've done
true_step = 0
report = False

#generate every
generate_every = 100
generate_template = tokenizer("EleutherAI is", return_tensors="pt").to(model_engine.local_rank)
clip_template = tokenizer("A drawing of a goose holding a knife<|CLIP|>", return_tensors="pt").to(model_engine.local_rank)
clip_idx_template = torch.sum(clip_template['attention_mask'], dim=-1).to("cpu").squeeze().item() - 1
text_so_far = list()

#Generate samples occasionally
def generate_from_template():
    if model_engine.local_rank == 0:
        out_embeds = model_engine(**clip_template, return_dict=True, output_hidden_states=True)['hidden_states'][-4].squeeze()
        CLIP_state = wrapper.proj(out_embeds[clip_idx_template]).cpu().detach().numpy()
        np.savetxt('clip.out', CLIP_state, delimiter=",")


for _, data_elem in pbar:
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
        start = data_elem['start']
        end = data_elem['end']
        last_layer = out_embeds[-4].squeeze() # -1 for second to last layer
        #Get predicted clip embedding. Grab from sequence_len dimension
        clip_embeds = torch.zeros((data.clip_batch_size, neo_hidden)).to(model_engine.local_rank)
        for i,j in enumerate(idx.tolist()):
            clip_embeds[i] = last_layer[i][j]

        #Project to the correct size
        clip_embeds = wrapper.proj(clip_embeds.to(torch.float16))
        #Frozen gradients for accum
        latent_copy = data_elem['latent_vecs']
        latent_copy[data_elem['start']:data_elem['end']] = clip_embeds

        latent_copy = latent_copy.to(torch.float32)
        latent_vecs = data_elem['latent_vecs'].to(torch.float32)

        #Compute contrastive loss
        if lschedule == "constant":
            actual_lambda = lambda_coeff
        elif lschedule == "truncated_sine":
            actual_lambda = lambda_coeff * max(0, math.sin(2*math.pi / lperiod * true_step))
        elif lschedule == "shifted_sine":
            actual_lambda = lambda_coeff * math.sin(math.pi / lperiod * true_step)**2
        else:
            raise NotImplementedError()    
        loss = actual_lambda * clip_loss(latent_copy,  latent_vecs,\
        temp_tensor, local_rank=model_engine.local_rank)

    else:
        #compute AR loss if Pile data
        n_text_toks = data_elem['clip_idx'].sum()
        loss = ar_loss(model_out, data_elem['input_ids'], local_rank=model_engine.local_rank) / n_text_toks

    if model_engine.is_gradient_accumulation_boundary():
        true_step += 1
        report=True
    #Accumulate loss, check if NaN. If not, update progress
    if not torch.any(loss.isnan()):
        model_engine.backward(loss.to(torch.float32))
        if data_elem['is_accum']:
            model_engine.step()
        loss_progress += loss.to(torch.float32).detach().cpu().item()
        loss_step_count += 1

        if data_elem['use_distill']:
            clip_loss_progress += loss.to(torch.float32).detach().cpu().item()
            clip_step_count += 1
        else:
            ar_loss_progress += loss.to(torch.float32).detach().cpu().item()
            ar_step_count += 1
    if true_step%generate_every==0:
        generate_from_template()
    
    #Update loss progress. For WandB
    if report:
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
        report=False
    #Save model
    if (true_step+1)%save_every==0:
        torch.distributed.barrier()
        ckpt_id = (true_step+1)

        # local or global?
        if model_engine.local_rank == 0:
            model_engine.module.lm.save_pretrained(f"models/{wandb.run.name}/{ckpt_id}", ckpt_id)
            torch.save(wrapper.proj, f"models/{wandb.run.name}/projection.pt")

model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)
tokenizer.save_pretrained("GPT-Neo-Enriched")