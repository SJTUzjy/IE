import torch
from transformers import AdamW
from utils.data import MyDataset
from model import *
from tqdm import tqdm
import os
Batch_size=256
model_name="hfl/chinese-xlnet-base"
#device=torch.device("cpu")
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_accuracy = 0.0

#set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(seed)
    random.seed(seed)

def train_eval(model,criterion,optimizer,train_dataloader,val_dataloader,epochs=100):
    model.to(device)
    print("----------begin to train---------")
    epoch_loss=0
    for epoch in range(epochs):
        model.train()
        epoch_loss=0
        print("epoch:%d"%(epoch))
        for i,batch in enumerate(tqdm(train_dataloader)):
            batch=tuple(t.to(device)for t in batch)
            logits=model(batch[0],batch[1],batch[2])
            loss=criterion(logits,batch[3])

            epoch_loss+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%100==0:
                eval(model,optimizer,val_dataloader,epoch)

#计算正确率
def flat_accuracy(pred,label):
    pred_flat=np.argmax(pred,axis=1).flatten()
    label_flat=label.flatten()
    return np.sum(pred_flat==label_flat)/len(label_flat)



def save(epoch,model,optimizer,accuracy):
    filepath=os.path.join('checkpoint_model_episode_{}_score_{}.pth'.format(epoch,accuracy))  #最终参数模型
    torch.save({'epoch':epoch,'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()},
               filepath)


def load(model,optimizer,checkpoint_path):

    model_ckpt=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(model_ckpt['state_dict'])
    optimizer.load_state_dict(model_ckpt['optimizer'])
    print("finish loading")
    return model,optimizer

def eval(model,optimizer,val_dataloader,epoch):
    model.eval()
    eval_accuracy,step=0,0
    for i,batch in enumerate(tqdm(val_dataloader)):
        batch=tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits=model(batch[0],batch[1],batch[2])
            logits=logits.detach().cpu().numpy()
            label=batch[3].cpu().numpy()
            tmp_eval_accuracy=flat_accuracy(logits,label)
            eval_accuracy+=tmp_eval_accuracy
            step+=1
    print("epoch:%d,test accuracy:%f"%(epoch,eval_accuracy/step))
    global best_accuracy
    if best_accuracy<eval_accuracy/step:
        best_accuracy=eval_accuracy/step
        save(epoch,model,optimizer,best_accuracy)



if __name__ == '__main__':
    set_seed(2)
    dataset = MyDataset('long_comments.csv')
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset,val_dataset=torch.utils.data.random_split(dataset,[train_size,val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=Batch_size,num_workers=0,shuffle=True)
    #token_idx,attn_masks,token_type_ids,label = train_dataloader[0]
    #print(token_idx,attn_masks,token_type_ids,label)
    # for token_idx,attn_masks,token_type_ids,label in train_dataloader:
    #         print(token_type_ids,attn_masks, label)
    #         break
    val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=Batch_size,num_workers=0,shuffle=False)

    # num_epoches = 1
    # for epoch in range(num_epoches):
    #     for token_idx,attn_masks,token_type_ids,label in train_dataloader:
    #         print(token_idx,attn_masks,token_type_ids, label)
    model=MyModel(freeze_bert=True,model_name=model_name,bert_hidden_size=768,num_class=5)
    criterion=nn.CrossEntropyLoss()
    optimizer=AdamW(model.parameters(),lr=1e-5,weight_decay=1e-2)

    train_eval(model,criterion,optimizer,train_dataloader,val_dataloader,epochs=100)
    