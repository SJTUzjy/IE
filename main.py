import torch
from transformers import BertModel, BertTokenizer
from utils.data import MyDataset

if __name__ == '__main__':
    """
    dataset = MyDataset('long_comments.csv')
    dataloader = torch.utils.data.DataLoader(dataset)
    num_epoches = 100
    for epoch in range(num_epoches):
        for sentence, label in dataloader:
            print(sentence, label)
    """