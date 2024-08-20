"""
@author: Joy Battocchio
Training script for attributes estimation model
Trains both the backbone and the heads
"""

import sys
# sys.path.append('RAFT/core')
# sys.path.append('/home/joy.battocchio/fake-video-detection/RAFT/core')
from sklearn.metrics import classification_report as CR
import torch
import torch.utils.data as data
from tqdm import tqdm
import argparse 
import os
import numpy as np
from glob import glob
from dataset import BL_dataset, Video_dataset
from model import Resnet3d, Resnet3d_18
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
import configparser
from loss import CustomLoss, RealDistLoss
from torch_kmeans import KMeans

configParser = configparser.RawConfigParser()   
configFilePath = "cuda.config"
configParser.read(configFilePath)
DEVICE = configParser.get("cuda-config", "device")

writer = SummaryWriter()

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

MODE = 'resnet18_3d'

SEED = 1234

os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type = str, default = None)
parser.add_argument('--epochs',type=int,default = 500)
parser.add_argument('--lr',type=float,default = 0.01)
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')


def save_checkpoint(epoch, net, optimizer):
    dir = f'{CURRENT_PATH}/checkpoints/{MODE}'
    os.makedirs(dir, exist_ok=True)
    torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(dir, f'epoch_{epoch}.pth'))

def evaluation(model, dataloader, loss_fn, e):
    cumulative_loss = 0

    acc = 0
    tr = 0
    tf = 0
    fr = 0
    ff = 0
    
    iters = 0
    total_y = torch.empty(0,device=DEVICE)
    total_pred = torch.empty(0,device=DEVICE)
    
    for i, (x, y) in enumerate(tqdm(dataloader, desc = 'validation', disable=True)):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        #if x.shape[0] == 1: break
        
        # inference
        if x.shape[0] == 1: x = x.unsqueeze(0)
        logit, _ = model(x)
        logit = logit.unsqueeze(0)
        loss = loss_fn.bce_loss(logit, y).mean()
            
        pred = torch.round(logit)
        total_pred = torch.cat((total_pred, pred))
        total_y = torch.cat((total_y, y))

        for p, l in zip(pred,y):
            if p == l:
                acc += 1
                if l == 1:
                    tr += 1
                else:
                    tf += 1
            else:
                if l == 1:
                    ff += 1
                else:
                    fr += 1
            
            iters +=1
        
        cumulative_loss += loss.item()
        # break
    
    print('-------Validation accuracy-------')
    print(f'Accuracy: \t{acc/iters}')
    print(f'True real: \t{tr}')
    print(f'True fake: \t{tf}')
    print(f'False real: \t{fr}')
    print(f'False fake: \t{ff}', flush=True)
    writer.add_scalar("Accuracy/eval", acc/iters, e)
    print(CR(total_y.cpu(),total_pred.cpu()))
    
    return cumulative_loss/iters
    
    
def train(model, dataloader_train, dataloader_val, optim, loss_fn, start, epochs, weights, resume):
    
    avg_loss_val = 0

    cluster_center = torch.empty(0,device=DEVICE)
    all_labels = torch.empty(0,device = DEVICE)

    def train_epoch(e, cluster_center):
        nonlocal all_labels
        nonlocal resume

        all_embeddings = torch.empty(0, device=DEVICE)
        cumulative_loss = 0
        
        acc = 0
        tr = 0
        tf = 0
        fr = 0
        ff = 0
        
        iters = 0
        
        for i, (x, y) in enumerate(tqdm(dataloader_train, desc = f'{epoch}', disable=True)):
            
            optimizer.zero_grad()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            if x.shape[0] == 1: break


            pred, emb = model(x)
            all_embeddings = torch.cat((all_embeddings, emb))
            
            loss_weights = torch.tensor([weights[int(label.item())] for label in y], device = DEVICE)
            
            if e == 0 or resume:
                all_labels = torch.cat((all_labels, y))
                loss = (loss_fn.bce_loss(pred, y)*loss_weights).mean()
            else:
                batch_distances = torch.cdist(emb, cluster_center)
                loss = loss_fn(pred, y, batch_distances, e, loss_weights)
            
            loss.backward()
            
            optim.step()
            
            cumulative_loss += loss
            
            pred = torch.round(pred)

            for p, l in zip(pred,y):
                if p == l:
                    acc += 1
                    if l == 1:
                        tr += 1
                    else:
                        tf += 1
                else:
                    if l == 1:
                        ff += 1
                    else:
                        fr += 1
                
                iters +=1
            # break

        real_embedding = all_embeddings[all_labels == 1].detach()
        cluster_center = real_embedding.mean(0).unsqueeze(0).detach()

        print('-------Training accuracy-------')
        print(f'Accuracy: \t{acc/iters}')
        print(f'True real: \t{tr}')
        print(f'True fake: \t{tf}')
        print(f'False real: \t{fr}')
        print(f'False fake: \t{ff}')
        writer.add_scalar("Accuracy/train", acc/iters, e)
        
        resume = False
        return cumulative_loss/iters, cluster_center

    for epoch in range(start, epochs):
        
        # model.eval()
        # with torch.no_grad():
        #     avg_loss_val = evaluation(model, dataloader_val, loss_fn, epoch)
        # break
        model.train(True)
        avg_loss, cluster_center = train_epoch(epoch, cluster_center)
        model.train(False)
        
        if (epoch+1) % 1 == 0:
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                
                print(f'Epoch: \t{epoch}')
                print('-------Training loss-------')
                print(f'Loss: \t{avg_loss}')
                writer.add_scalar("Loss/train", avg_loss, epoch)
                
                avg_loss_val = evaluation(model, dataloader_val, loss_fn, epoch)
                #avg_loss_val = 0
                
                print('-------Validation loss-------')
                print(f'Loss: \t{avg_loss_val}')
                writer.add_scalar("Loss/eval", avg_loss_val, epoch)

            save_checkpoint(epoch+1, model, optimizer)

        # if (epoch+1) % 50 == 0:
        #     torch.cuda.empty_cache()
            
        #     with torch.no_grad():

        #         dataset_compression = '640x360_gen'
        #         technique = ['SEINE']
        #         dataset = Video_dataset(split='val', dataset = dataset_compression, small = None, augment=True, techniques=technique)
        #         dataloader = data.DataLoader(dataset, batch_size = 16, num_workers= 8, pin_memory=True)
        #         print("####### 50 epoch VALIDATION #######")
        #         print(dataset_compression, technique)
        #         _ = evaluation(model, dataloader, loss_fn, epoch)
        #         print("##############")
                



if __name__ == '__main__':

    args = parser.parse_args()
    ckp_path = args.checkpoint
    epochs = args.epochs
    lr = args.lr
    dataset_compression = '640x360_gen'
    technique = ['DynamiCrafter', "TokenFlow"]
    dataset_train = Video_dataset(split='train', dataset = dataset_compression, small = None, augment=True, techniques=technique)
    dataset_val = Video_dataset(split='val', dataset = dataset_compression, small = None, augment=False, techniques=technique)
    
    dataloader_train = data.DataLoader(dataset_train, batch_size = 16, num_workers= 8, pin_memory=True)
    dataloader_val = data.DataLoader(dataset_val, batch_size = 1, num_workers=8, pin_memory=True)
    
    writer.add_text(MODE, 'Resnet50_3d augmented jpg lr 0.001')

    model = Resnet3d_18()

    # checkpoint = torch.load('/home/joy.battocchio/fake-video-detection/checkpoints/resnet50_3d_jan/epoch_100.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model.eval()
    model.to(DEVICE)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    #ckp_path = "/home/joy.battocchio/fake-video-detection/checkpoints/resnet18_3d/checkpoint.pth"

    resume = False

    if ckp_path is not None:
        resume = True
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])   
        start = checkpoint['epoch']
        print('resuming from epoch {}'.format(start))
    else:
        start = 0
    print(start, flush=True)
    real_size = len([x for x in dataset_train.data if 'clips_original' in x])
    fake_size = dataset_train.__len__() - real_size
    total_size = fake_size + real_size
    weights = torch.tensor([real_size/total_size, fake_size/total_size], device = DEVICE)
    print(weights)
    loss_fn = RealDistLoss()  
    
    train(model,dataloader_train, dataloader_val, optimizer, loss_fn, start, epochs, weights, resume)
    writer.flush()
    writer.close()
    print(dataset_compression)