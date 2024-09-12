import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ['HF_TOKEN'] = "huggingface에서 발급한 토큰 입력"

tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b-it')

input_text = "너에 대해서 설명해 줄래?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))