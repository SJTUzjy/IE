from torch.utils.data import Dataset, DataLoader, Sampler
import csv
from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, filename,max_len=100,model_name="hfl/chinese-xlnet-base"):
        self.data = []
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            t = 0
            for row in reader:
                t += 1
                if t == 1: continue
                self.data.append(row)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.max_len=max_len

    def __getitem__(self, idx):
        tmp = self.data[idx]
        encoder_pair=self.tokenizer(tmp[3],
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors="pt"
                                    )
        token_idx=encoder_pair["input_ids"].squeeze(0)  #tensor of token ids
        attn_masks=encoder_pair["attention_mask"].squeeze(0)    #binary tensor with "0" for padded value and "1" for other value
        token_type_ids=encoder_pair["token_type_ids"].squeeze(0)    #binary tensor with "0" for the 1st sentence token and "1" for the 2nd sentence tokens,and "3" for padding tokens
        #print(tmp[3])
        label_5 = int(tmp[6])-1
        if int(tmp[6]) > 3: label_2 = float(1)
        elif int(tmp[6]) < 3: label_2 = float(0)
        return token_idx, attn_masks, token_type_ids, label_2

    def __len__(self):
        return len(self.data)


