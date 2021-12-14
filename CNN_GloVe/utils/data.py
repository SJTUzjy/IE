from gensim.models.keyedvectors import load_word2vec_format
from torch.utils.data import Dataset
import numpy as np 

# def makeReadable(path):
#     with open(path, 'r+', encoding='utf-8') as f:
#         num_lines = len(f.readlines())
#         gensim_first_line = '%d %d\n' % (num_lines, DIMENSIONS)
#         old = f.read()
#         f.seek(0)
#         f.write(gensim_first_line)
#         f.write(old)

class MyDataset(Dataset):
    def __init__(self, dataset_path, embedding_path):
        self.rating = []
        self.comment = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            for row in data:
                self.rating.append(int(row[0]))
                self.comment.append(row[2:])

        self.embedding = load_word2vec_format(embedding_path, encoding='utf-8', binary=False)

    def __getitem__(self, idx):
        label = float(self.rating[idx])
        words = self.comment[idx].split()
        input = np.zeros((100, 50))
        index = 0
        for word in words[:100]:
            try:
                input[index] = self.embedding[word]
            except KeyError:
                input[index] = self.embedding['<unk>']

        # swapaxes: (words, dim) -> (dim, words)
        # ... as conv1d works on the last axis.
        return input.swapaxes(0, 1), label

    def __len__(self):
        return len(self.rating)

