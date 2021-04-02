from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2Tokenizer, GPTNeoConfig
from torch.utils.data import IterableDataset, DataLoader 
from lm_dataformat import *
import torch

#initalize a smol boi
config = GPTNeoConfig(hidden_size = 128, num_layers = 24, attention_layers = 24)
#create model
model = GPTNeoForCausalLM(config)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

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
            img_latents.append([[0]*512])
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
        print(txts)
        toks = tok.batch_encode_plus(txts, max_length=2048, truncation=True, padding=True, return_tensors="pt")
        #Get the index of the clip tokens.
        clip_idx = torch.sum(toks.attention_mask, dim=-1) - torch.tensor([1] * len(txts))
        #Get latent vectors
        latents = torch.stack([torch.tensor(x) for x in img_latents])

        cc = self.cur_clip
        #Flip cur clip
        self.cur_clip = not self.cur_clip 
        return {
            'toks' : toks,
            'latent_vecs' : latents,
            'clip_idx' : clip_idx, 
            'use_distill' : cc,
        }



    
data = DistillDataset(tokenizer = tokenizer, clip_batch_size = 128,\
    clip_dataset_dir = "../clip_latents_100k.jsonl.zst",\
    pile_dataset_dir = "../val.jsonl.zst")
for elem in data:
    print(elem)
    break