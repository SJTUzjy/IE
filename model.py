from torch import nn
from transformers import AutoModel,AutoModelForSequenceClassification
import torch
import numpy as np
import random

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#初步定义模型，需要修改完善
class MyModel(nn.Module):
    def __init__(self,freeze_bert=False,model_name="hfl/chinese-xlnet-base",bert_hidden_size=768,num_class=5,lstm_hidden_dim=384,n_layers=2,bidirectional=True):
        super(MyModel,self).__init__()
        self.n_layers=n_layers
        self.hidden_dim=lstm_hidden_dim
        self.bidirectional=bidirectional
        #output_hidden_state=true 才可获得hidden_states
        self.bert=AutoModel.from_pretrained(model_name)
        #self.bert=AutoModelForSequenceClassification.from_pretrained(model_name,num)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad=False

        self.lstm=nn.LSTM(bert_hidden_size,lstm_hidden_dim,n_layers,batch_first=True,bidirectional=bidirectional)

        self.dropout=nn.Dropout(0.5)
        if bidirectional:
            self.layer1=nn.Sequential(
                nn.Linear(lstm_hidden_dim*2,128),
                nn.ReLU()
            )
        else:
            self.layer1=nn.Sequential(
                nn.Linear(lstm_hidden_dim,128),
                nn.ReLU()
            )

        self.layer2=nn.Linear(128,num_class)
        # self.fc=nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.Linear(bert_hidden_size*4,num_class)
        # )


    def forward(self,input_ids,attn_masks,token_type_ids):
        output=self.bert(input_ids,token_type_ids=token_type_ids,attention_mask=attn_masks)
        #获得后面四层的hidden-state的输出进行拼接，shape[batch-size,seq-len,bert-hidden-size*4]
        hidden_states=output.last_hidden_state
        #print(hidden_states.shape)
        lstm_out,(hidden_last,cn_last)=self.lstm(hidden_states,None)
        if self.bidirectional:
            hidden_last_L=hidden_last[-2]
            hidden_last_R=hidden_last[-1]
            hidden_last_out=torch.cat([hidden_last_L,hidden_last_R],dim=-1)
        else:
            hidden_last_out=hidden_last[-1]
        out=self.dropout(hidden_last_out)
        out=self.layer1(out)
        out=self.layer2(out)
        return out
        #hidden_states=torch.cat( tuple( [ output.hidden_states[i] for i in [-1,-2,-3,-4] ]  ) ,dim=-1 )
        #first_hidden_states=hidden_states[:,0,:]#选取第一个词的hidden-state作为最后的输出
        #logits=self.fc(first_hidden_states)
        #return output



