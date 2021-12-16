import os
import random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AdamW
from model import MyModel
from utils.data import MyDataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET = 'comments_cleaned.txt'
VECTOR = 'glove_vectors.txt'
BATCH_SIZE = 64
DIMENSIONS = 50

best_accuracy = 0.0

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

def flat_accuracy(pred,label):
    pred_flat = pred.flatten()
    label_flat = label.flatten()
    return np.sum(pred_flat == label_flat) / len(label_flat)

def load(model, optimizer, checkpoint_path):
    model_ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(model_ckpt['state_dict'])
    optimizer.load_state_dict(model_ckpt['optimizer'])
    print("Finish loading!")
    return model, optimizer

def save(epoch, model, optimizer, accuracy):
    filepath = os.path.join('checkpoint_model_GloVe_episode_{}_score_{}.pth'.format(epoch, accuracy))
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filepath)

def train_eval(model, criterion, optimizer, train_dataloader, valid_dataloader, epochs=100):
    model.to(DEVICE)
    criterion.to(DEVICE)
    print('----- This is CNN-GloVe model! Begin to train -----')
    epoch_loss = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print('┏Epoch #%d' % epoch)
        for i, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t.to(DEVICE) for t in batch)
            logits = model(batch[0])
            loss = criterion(logits, batch[1])
            epoch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        eval(model, optimizer, valid_dataloader, epoch)

def eval(model, optimizer, valid_dataloader, epoch):
    global best_accuracy
    model.eval()
    eval_accuracy, step = 0, 0
    for i, batch in enumerate(tqdm(valid_dataloader)):
        batch = tuple(t.to(DEVICE) for t in batch)
        with torch.no_grad():
            logits = model.predict(batch[0])
            logits = logits.detach().cpu().numpy()
            label = batch[1].cpu().numpy()
            eval_accuracy += flat_accuracy(logits, label)
            step += 1
    print('┗Epoch #%d, validation accuracy:%f' % (epoch, eval_accuracy / step))
    if best_accuracy < eval_accuracy / step:
        best_accuracy = eval_accuracy / step
        save(epoch, model, optimizer, best_accuracy)


if __name__ == '__main__':
    set_seed(42)
    dataset = MyDataset(DATASET, VECTOR)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)

    model = MyModel()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    train_eval(model, criterion, optimizer, train_dataloader, valid_dataloader, 100)