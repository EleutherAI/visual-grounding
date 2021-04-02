from transformers import GPTNeoModel, GPTNeoForCausalLM,\
    GPT2Tokenizer, GPTNeoConfig

#initalize a smol boi
config = GPTNeoConfig(hidden_size = 128, num_layers = 24, attention_layers = 24)
#create model
model = GPTNeoForCausalLM(config)

print(len(model))