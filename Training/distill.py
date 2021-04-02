from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPTNeoTokenizer, GPTNeoConfig
from torch.utils.data import Datasetl, DataLoader 
from lm_dataformat import *

#initalize a smol boi
config = GPTNeoConfig(hidden_size = 128, num_layers = 24, attention_layers = 24)
#create model
model = GPTNeoForCausalLM(config)

print(len(model))

#pytorch dataset for clip juicing
class gimmiedatclipjuice(Dataset):
    def __init__(self,\
        tokenizer, clip_batch_size,
        clip_dataset_dir, pile_dataset_dir, steps = 1e6):
        self.clip_dataset_dir = clip_dataset_dir
        self.pile_dataset_dir = pile_dataset_dir

        self.clip_rdr = Reader(self.clip_dataset_dir)
        self.pile_rdr = Reader(self.pile_dataset_dir)

        #Steps is the total number of elements we should use. Half from CLIP, half from AR
        self.steps = steps
        #How many elements are in a single contrastive clip batch
        self.clip_batch_size = clip_batch_size

        #Start on an example of the pile
        self.cur_clip = False
    def __len__(self):
        return self.steps
    def __getitem__(self):
        #Return an element from the pile
        if not self.cur_clip:


    
    