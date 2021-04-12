from utility import *
import torch
from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2TokenizerFast, GPTNeoConfig, AdamW
from torch.utils.data import IterableDataset, DataLoader
from auto_tqdm import tqdm

#conf = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-1.3B")
#model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", config=conf)
#model.resize_token_embeddings(len(tokenizer)+2)


#projection=projection_model(neo_hidden=model.config.hidden_size)
#projection.half()

#wrapper = ModelWrapper(model, projection)


data = DistillDataset(tokenizer = tokenizer, clip_batch_size = clip_bs,\
    clip_dataset_dir = "../../clip/",\
    pile_dataset_dir = "../../pile/", local_rank="cuda")

loader = DataLoader(dataset=data, batch_size=1)
pbar = tqdm(enumerate(data), total=len(data))


for batch, data_elem in pbar:
    print(data_elem)
    pass
    if batch == 10:
        break