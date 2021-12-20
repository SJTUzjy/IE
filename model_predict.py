import xml.dom.minidom
from transformers import AdamW
from torch import nn
from transformers import AutoModel,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
import random
from model import MyModel
from transformers import AutoTokenizer

model_name = "hfl/chinese-xlnet-base"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load(model,optimizer,checkpoint_path):

    model_ckpt=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(model_ckpt['state_dict'])
    optimizer.load_state_dict(model_ckpt['optimizer'])
    print("finish loading")
    return model,optimizer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model=MyModel(freeze_bert=False,model_name=model_name,bert_hidden_size=768,num_class=2)
    model = model.to("cuda:0")
    optimizer=AdamW(model.parameters(),lr=1e-5,weight_decay=1e-2)
    model, optimizer = load(model, optimizer, "./checkpoint_model_episode_2_score_0.8907471124008735.pth")
    domInput = xml.dom.minidom.parse('task2input.xml')
    rootInput = domInput.documentElement
    domOutput = xml.dom.minidom.getDOMImplementation().createDocument(None, rootInput.nodeName, None)
    rootOutput = domOutput.documentElement
    elementsInput = rootInput.getElementsByTagName('weibo')
    for element in elementsInput:
        element1 = element
        sentence = element.childNodes[0].nodeValue
        print(sentence)
        encoder_pair= tokenizer(sentence,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=100,
                                    return_tensors="pt"
                                    )
        token_idx=encoder_pair["input_ids"].to("cuda:0") #tensor of token ids
        attn_masks=encoder_pair["attention_mask"].to("cuda:0")   #binary tensor with "0" for padded value and "1" for other value
        token_type_ids=encoder_pair["token_type_ids"].to("cuda:0")
        result = model.predict(token_idx, attn_masks, token_type_ids)
        element1.setAttribute('polarity', str("1" if result == 1 else "-1"))
        rootOutput.appendChild(element1)
    with open('task2output.xml', 'w', encoding='utf-8') as f:
        domOutput.writexml(f, addindent='\t', newl = '\n', encoding = 'utf-8')
