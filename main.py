import torch
from transformers import BertModel, BertTokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).cuda()
input_text = "Here is some text to encode"
input_ids = tokenizer.encode(input_text, add_special_tokens=False)
input_ids = torch.tensor([input_ids]).cuda()
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]