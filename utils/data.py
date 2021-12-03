from torch.utils.data import Dataset, DataLoader, Sampler
import csv
from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, filename,max_len=150,model_name="hfl/chinese-xlnet-base"):
        self.data = []
        f = open(filename, 'r', encoding='utf-8')
        with f:
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
        token_idx=encoder_pair["input_ids"].squeeze(0)
        attn_masks=encoder_pair["attention_mask"].squeeze(0)
        token_type_ids=encoder_pair["token_type_ids"].squeeze(0)
        print(tmp[3])
        return token_idx,attn_masks,token_type_ids,int(tmp[6])

    def __len__(self):
        return len(self.data)


