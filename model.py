from torch import nn
from transformers import AutoModel
import torch
import numpy as np
import random

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#初步定义模型，需要修改完善
class MyModel(nn.Module):
    def __init__(self,freeze_bert=False,model_name="hfl/chinese-xlnet-base",bert_hidden_size=768,num_class=5):
        super(MyModel,self).__init__()
        #output_hidden_state=true 才可获得hidden_states
        self.bert=AutoModel.from_pretrained(model_name,output_hidden_states=True,return_dict=True)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad=False


        self.fc=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(bert_hidden_size*4,num_class)
        )

    def forward(self,input_ids,attn_masks,token_type_ids):
        output=self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attn_masks)
        #获得后面四层的hidden-state的输出进行拼接，shape[batch-size,seq-len,bert-hidden-size*4]
        hidden_states=torch.cat( tuple( [ output.hidden_states[i] for i in [-1,-2,-3,-4] ]  ) ,dim=-1 )
        first_hidden_states=hidden_states[:,0,:]#选取第一个词的hidden-state作为最后的输出
        logits=self.fc(first_hidden_states)
        return logits



