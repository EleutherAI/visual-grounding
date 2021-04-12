import torch
import deepspeed
import wandb
import os
from random import randint
import math
from lm_dataformat import *

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


class ModelWrapper(torch.nn.Module):
    def __init__(self, lm, proj):
        super(ModelWrapper, self).__init__()
        self.lm = lm
        self.proj = proj
    def forward(self, **kwargs):
        return self.lm(**kwargs)


class ContrastiveLossHandler(IterableDataset):
    def __init__(self, reader, tokenizer,
    text_len=128, special_token="<|CLIP|>", micro=20, macro=1.6e4):
        super(ContrastiveLossHandler, self).__init__()
        self.micro=micro
        self.macro=macro
        self.text_len=text_len
        self.special_token=special_token
        self.text_list=list()
        self.latent=None
        #Have we already accumulated a list
        self.is_accum = False
        self.step = 0
        self.reader=reader
        self.tokenizer=tokenizer
    def __iter__(self):
        return self
    def __next__(self):
        if self.is_accum:
            out_text = self.text_list[self.step*self.micro:min((self.step+1)*self.micro, self.macro)]
            out_latents = self.latents
            #If we have reached the end
            if min((self.step+1)*self.micro, self.macro) == self.macro:
                self.is_accum = False
            else:
                self.step += 1
            return {
                **out_text,
                'latent_vecs':out_latents,
                'is_accum':self.is_accum,
            }
        else:
            #If we have nothing accumulated, accum now
            self.pre_accumulate()
            #Then return the first set
            return next(self)
    #Loads a macro_batch of contrastive examples
    def pre_accumulate(self):
        self.is_accum = True

        txts=list()
        latents=list()
        #Read from the streamer
        for _ in range(self.macro):
            txt, img_latent=next(self.reader)
            #text split
            ts = txt.split()
            if len(ts) > self.text_len:
                #Plus one so that we can have our special token
                start = randint(0, len(ts)-(self.text_len+1))
                #End of the string should be text_len-1 distance from start at most
                end = min(start + (self.text_len) - 1, len(ts) - start)
                txt = " ".join(ts[start:end])
            txt+=self.special_token
            txts.append(txt)
            latents.append([img_latent])
        #Tokenize text
        toks = tok.batch_encode_plus(txts, max_length=self.text_len, truncation=True,\
            padding="max_length", return_tensors="pt")
        self.text_list=toks
        self.latents=torch.cat([torch.tensor(x) for x in latents], dim=0)

#pytorch dataset for clip juicing
class DistillDataset(IterableDataset):
    def __init__(self,\
        tokenizer, clip_batch_size,
        clip_dataset_dir, pile_dataset_dir,
        special_token = "<|CLIP|>", steps = 1e6, local_rank):
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
        self.local_rank=local_rank

        #Store previous image latents. Used to accumulate larger batch sizes
        self.last_latents=None
        #Are we currently in a contrastive batch where we need to accumulate loss?
        self.currently_contrastive=False
        self.closs_handler=ContrastiveLossHandler(reader=clip_rdr,tokenizer=self.tokenizer)
        
    def __len__(self):
        return int(self.steps)
    def __iter__(self):
        return self
    def __next__(self):
        tok = self.tokenizer
        txts = list()
        img_latents = list()

        use_clip = False

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
                    end = min(start + 1024, len(ts) - start)
                    text = " ".join(ts[start:])
                txts.append(text)

            #Tokenize text
            toks = tok.batch_encode_plus(txts, max_length=1024, truncation=True,\
                padding="max_length", return_tensors="pt").to(self.local_rank)
            img_latents.append([[0]*clip_hidden])

            #Mixing
            self.mix_step += 1
            #Get latent vectors
            latents = torch.cat([torch.tensor(x) for x in img_latents], dim=0).to(self.local_rank)
        
        #Return an element from CLIP
        else:
            data=next(self.closs_handler)
            #If we finished closs
            if not data['is_accum']:
                self.mix_step+=1
            toks={
                'input_ids':data['input_ids'],
                'attention_mask':data['attention_mask'],
            }
            toks.to(self.local_rank)
            latents=data['latent_vecs']
            latents.to(self.local_rank)

        #Get the index of the clip tokens.
        clip_idx = (torch.sum(toks.attention_mask, dim=-1).to("cpu") - torch.tensor([1] * len(txts)))

        return {
            **toks,
            'latent_vecs' : latents,
            'clip_idx' : clip_idx, 
            'use_distill' : use_clip,
        }







