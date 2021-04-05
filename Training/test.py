#Used to see what the actual fuck is causing our memory leak ;-;
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
from transformers import Trainer, TrainingArguments

#Da models
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

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
    def __len__(self):
        return int(self.steps)
    def __iter__(self):
        return self
    def __next__(self):
        tok = self.tokenizer
        txts = list()
        img_latents = list()
    
        text, _ =next(self.pile_rdr)
        txts.append(text)

        #Tokenize text
        toks = tok.batch_encode_plus(txts, max_length=2048,\
                truncation=True, padding="max_length",\
                return_tensors="pt")
        tgt = toks['input_ids'][:, 1:]


        return {
            **toks,
            'labels':tgt,
        }

data = DistillDataset(tokenizer = tokenizer, clip_batch_size = 1,\
    clip_dataset_dir = "../../../clip_latents_100k.jsonl.zst",\
    pile_dataset_dir = "../../../val.jsonl.zst")

training_args = TrainingArguments(output_dir='./results',
                                  num_train_epochs=1,
                                  logging_steps=5000,
                                  save_steps=5000,                                   
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  warmup_steps=100,
                                  weight_decay=0.01,  
                                  logging_dir='./logs')
trainer = Trainer(model=model, args=training_args,  
                  train_dataset=data,
                  eval_dataset=data)

trainer.train()
