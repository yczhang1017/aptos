
import os
import numpy as np
import pandas as pd

import PIL
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from torchvision.transforms import transforms
import torch.backends.cudnn as cudnn
import time
import argparse
import pretrainedmodels
from sklearn.model_selection import train_test_split
import torch.nn.functional as f
from nasnetv2 import nasnetv2
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from kappas import quadratic_weighted_kappa
from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F

#from torch.autograd import Variable

#import torch.utils.model_zoo as model_zoo



parser = argparse.ArgumentParser(
    description='train a model')
parser.add_argument('--root', default='./',
                    type=str, help='directory of the data')
parser.add_argument('--batch', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('-e','--epochs', default=48, type=int,
                    help='number of epochs to train')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('--resume', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--model', default='pnasnet5large', type=str,
                    help='model name')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from') 
parser.add_argument('--size', default=224, type=int,
                    help='image size')
parser.add_argument('--print', default=10, type=int,
                    help='print freq')
parser.add_argument('--loss', default='wmse',  choices=['mse', 'wmse','huber','l1_cut'], type=str,
                    help='type of loss')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

mean=[0.4402, 0.2334, 0.0674]
std=[0.2392, 0.1326, 0.0470]
transform= { 
 'train':transforms.Compose([
     transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
     transforms.RandomResizedCrop(args.size,scale=(0.2, 1.0), 
                                  ratio=(0.9, 1.1111),interpolation=PIL.Image.BILINEAR),
     transforms.ColorJitter(0.2,0.1,0.1,0.04),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ]),      
 'val':transforms.Compose([
     transforms.Resize((args.size,args.size),
                       interpolation=PIL.Image.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ])}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings



class APTOSDataset(Dataset):
    def __init__(self, phase, data ,transform):
        self.phase=phase
        self.data=data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.phase in ['train','val']:
            x, y = self.data[idx]
        elif self.phase == 'test' :
            x = self.data[idx]
        
        if '_' in x:
            root='blind512'
        else:
            root='train512'
        
        img_name = os.path.join(root,
                                x + '.jpg')
        image = PIL.Image.open(img_name)
        '''
        w,h = image.size
        a= np.sqrt(w*h)
        tf = transforms.Compose([
                transforms.RandomRotation(12),
                transforms.CenterCrop((a,a))])
        image =tf(image)    
        '''    
        image = self.transform(image)
        if self.phase in ['train','val']:
            return image, y
        elif self.phase == 'test' :
            return image 
'''        
class MSELogLoss(nn.Module):
    def __init__(self):
        super(MSELogLoss, self).__init__()
    def forward(self, inputs, targets):
        return torch.mean(-(1-(inputs-targets)*(inputs-targets)*(1/4.5/4.5)).log())

class M4Loss(nn.Module):
    def __init__(self):
        super(M4Loss, self).__init__()
    def forward(self, inputs, targets):
        return torch.mean((inputs-targets)*(inputs-targets)*(inputs-targets)*(inputs-targets))
'''
class weighted_mse(nn.Module):
    def __init__(self, weight):
        super(weighted_mse, self).__init__()
        self.weight=weight.float().to(device)
    def forward(self, input, target):
        truth=target.long()
        return torch.mean(self.weight[truth]*(input-target)*(input-target))
    
class L1_cut_loss(nn.Module):
    def __init__(self, weight):
        super(L1_cut_loss, self).__init__()
        self.weight=weight.float().to(device)
    def forward(self, input, target):
        truth=target.long()
        loss=self.weight[truth]*f.relu(torch.abs(input-target)-0.5)
        return loss.mean()

    
def main():
    train_csv=os.path.join(args.root, 'train.csv')
    df  = pd.read_csv(train_csv)
    #dist= df.groupby('diagnosis').count().values.reshape(5)
    
    data={'train':None,'val':None}
    dataset={'train':None,'val':None}
    dataloader={'train':None,'val':None}
    data['train'], data['val'] = \
        train_test_split(df.values.tolist(), test_size=0.05, random_state=42)  
        
    ext_csv = os.path.join(args.root, 'exter-resized', 'trainLabels_cropped.csv')
    df2  = pd.read_csv(ext_csv, header=1 ,names = ['0','1','id_code', 'diagnosis']).iloc[:,2:4]
    df2['diagnosis'] = df2['diagnosis'].astype(int)
    data['train'] += df2.values.tolist()
    df=df.append(df2)
    print(df.groupby('diagnosis').count())
    
    print(len(data['train']),len(data['val']))
    dataset={x: APTOSDataset(x, data[x], transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: DataLoader(dataset[x],
            batch_size=args.batch, shuffle = (x=='train'),
            num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    
    elif args.loss == 'wmse':
        #weight = torch.pow(torch.tensor(dist[0]/dist,dtype=torch.float),0.4)
        #weight[-1]=weight.max()
        weight = torch.tensor([1,1.4,1.2,3,6])
        print(weight)
        criterion = weighted_mse(weight)
        
    elif args.loss== 'huber':
        criterion = nn.SmoothL1Loss()
        
    elif args.loss== 'l1_cut':
        #weight = torch.pow(torch.tensor(dist[0]/dist,dtype=torch.float),1/4)
        #weight[-1]=weight.max()
        weight = torch.tensor([1,1.4,1.2,3,6])
        print(weight)
        criterion = L1_cut_loss(weight)
    
    if args.model in pretrainedmodels.__dict__.keys():
        model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Sequential( 
                nn.BatchNorm1d(4320),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=4320, out_features=600, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(600),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=600, out_features=100, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(100),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=100, out_features=1, bias=True),
                )
    elif args.model == 'nasnetv2':
        model = nasnetv2()
    
    #print(model)
    model = model.to(device)
    if torch.cuda.is_available():
        model=nn.DataParallel(model)
        cudnn.benchmark = True    
    if args.checkpoint:
        print('Resuming training from epoch {}, loading {}...'
              .format(args.resume,args.checkpoint))
        weight_file=os.path.join(args.root,args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))    
 
    optimizer = optim.SGD(model.parameters(),lr=args.lr, 
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[16,24,32,40], gamma=0.1)
   
    for i in range(args.resume):
        scheduler.step()
    for epoch in range(args.resume,args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            nb = 0
            num = 0
            running_loss = 0
            running_correct = 0
            predict=[]
            truth=[]
            
            for inputs,targets in dataloader[phase]:
                t1 = time.time()
                batch = inputs.size(0)
                inputs = inputs.to(device)                
                targets= targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).reshape(batch)
                    loss = criterion(outputs, targets.float())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                nb +=1
                num += batch
                loss = loss.item() 
                running_loss += loss * inputs.size(0)
                propose=outputs.round().long().clamp(0,4)
                correct = (propose==targets).sum().item()
                acc = correct/batch*100
                running_correct +=correct
                p=propose.cpu().tolist()
                t=targets.cpu().tolist()
                predict+=p
                truth+=t
                t2 = time.time()
                if nb %args.print ==0:
                    
                    print('|'.join(str(x) for x in p))
                    print('|'.join(str(x) for x in t))
                    print('n:{:d}, l:{:.4f}|{:.4f}, a:{:.4f}|{:.4f}, t:{:.4f}' \
                          .format(num, loss, running_loss/num, acc, running_correct/num*100, t2-t1))
            
            print('num:', num, len(truth))
            cm = confusion_matrix(truth, predict, labels=[0,1,2,3,4])
            ht=histogram(truth,0,4)
            hp=histogram(predict,0,4)
            hm = np.outer(ht,hp)/np.float(num)
            kappa = cohen_kappa_score(truth, predict, labels=[0,1,2,3,4])
            kappa2= quadratic_weighted_kappa(truth, predict, 0, 4)
            print('='*5,phase,'='*5)
            print("Confusion matrix")
            print(cm)
            print("Hist matrix")
            print(ht)
            print(hp)
            print(hm)
            print('{:s}:{:d}, n:{:d}, l:{:.4f}, a:{:.4f}, k:{:.4f}, k2:{:.4f}, t:{:.4f}' \
                  .format(phase, epoch+1, num, running_loss/num, \
                          running_correct/num*100, kappa, kappa2, t2-t1))
            
            print('='*15)
            
            
            if phase == 'val':
                torch.save(model.state_dict(),
                          os.path.join(args.save_folder,'out_'+str(epoch+1)+'.pth'))
        print()
        
if __name__ == '__main__':
    main()        


