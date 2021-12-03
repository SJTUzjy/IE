import torch
from transformers import BertModel, BertTokenizer
from utils.data import MyDataset

if __name__ == '__main__':
    
    dataset = MyDataset('long_comments.csv')
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset)
    #token_idx,attn_masks,token_type_ids,label = train_dataloader[0]
    #print(token_idx,attn_masks,token_type_ids,label)
    for token_idx,attn_masks,token_type_ids,label in train_dataloader:
            print(token_idx.shape, label)
            break
    # test_dataloader=torch.utils.data.DataLoader(test_dataset)
    # num_epoches = 1
    # for epoch in range(num_epoches):
    #     for token_idx,attn_masks,token_type_ids,label in train_dataloader:
    #         print(token_idx,attn_masks,token_type_ids, label)
    
    