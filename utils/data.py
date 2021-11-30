from torch.utils.data import Dataset, DataLoader, Sampler
import csv

class MyDataset(Dataset):
    def __init__(self, filename):
        self.data = []
        f = open(filename, 'r', encoding='utf-8')
        with f:
            reader = csv.reader(f)
            t = 0
            for row in reader:
                t += 1
                if t == 1: continue
                self.data.append(row)
    def __getitem__(self, idx):
        tmp = self.data[idx]
        return [tmp[3], int(tmp[6])]

    def __len__(self):
        return len(self.data)

