from utility import *
import torch
from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2TokenizerFast, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader
import auto_tqdm as tqdm

tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", config=conf)
model.resize_token_embeddings(len(tokenizer)+2)


projection=projection_model()
projection.half()

wrapper = ModelWrapper(model, projection)


data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../clip/",\
    pile_dataset_dir = "../../pile/", local_rank="cuda")

loader = DataLoader(dataset=data, batch_size=1)
pbar = tqdm(enumerate(loader), total=len(data))


for batch, data_elem in pbar:
    pass
    if batch == 100:
        break