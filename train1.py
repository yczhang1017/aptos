
import os
import numpy as np
import pandas as pd

from PIL import Image
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
parser.add_argument('--size', default='256,320', type=str,
                    help='image size')
parser.add_argument('--print', default=10, type=int,
                    help='print freq')
parser.add_argument('--loss', default='mse',  choices=['mse', 'wmse','huber','l1_cut'], type=str,
                    help='type of loss')
parser.add_argument('--dataset', default='train640,prev640,IEEE640,messidor640', type=str,
                    help='previous competition dataset directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

mean=[0.4402, 0.2334, 0.0674]
std=[0.2392, 0.1326, 0.0470]
size = [int(i) for i in args.size.split(',')]
size =tuple(size) 

transform= { 
 'train':transforms.Compose([
     transforms.RandomRotation(10, resample=Image.BILINEAR),
     transforms.RandomResizedCrop(size,scale=(0.2, 1.0), 
                                  ratio=(1, 1.5),interpolation=Image.BILINEAR),
     transforms.ColorJitter(0.2,0.1,0.1,0.04),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ]),      
 'val':transforms.Compose([
     transforms.CenterCrop((512,640)),
     transforms.Resize(size,
                       interpolation=Image.BILINEAR),
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
        self.data_path = args.dataset.split(',')
        self.weights = [1, 0.8, 1, 0.9]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.phase in ['train','val']:
            row = self.data.iloc[idx]
            d = int(row['dataset'])
            root = self.data_path[d]
            y = float(row['diagnosis'])
            if d==3:
                y=(y+0.5)/4*5-0.5
            
        elif self.phase == 'test' :
            row = self.data.iloc[idx]
            root = self.data_path[0]
        
        img_name = os.path.join(root,
                                row['id'] +'.jpeg')
        image = Image.open(img_name)
        image = self.transform(image)
        if self.phase in ['train','val']:
            return image, y
        elif self.phase == 'test' :
            return image 

    
def main():
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.model in pretrainedmodels.__dict__.keys():
        model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
        model.avg_pool = nn.Conv2d(in_channels=4320, out_channels=4320, groups=4320, 
            kernel_size=(4,5), stride=1, bias=False)
            
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
                nn.Linear(in_features=100, out_features=5, bias=True),
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
    
    
    train_csv=os.path.join(args.root, 'train.csv')
    df1 = pd.read_csv(train_csv, header=1, names = ['id', 'diagnosis'], 
                            dtype={'id':str, 'diagnosis':np.int8})
    df1['dataset'] = 0
    df1, df_val = \
        train_test_split(df1, test_size=0.05, random_state=40)
    
    print('Current Competition:')
    print(df1.groupby('diagnosis').count())    
    
    #Previous dataset    
    ext_csv = os.path.join(args.root, 'exter-resized', 'trainLabels_cropped.csv')
    df2  = pd.read_csv(ext_csv,header=1,names = ['id','diagnosis'],
                       usecols=[2,3], dtype={'id':str, 'diagnosis':np.int8})
    df2['diagnosis'] = df2['diagnosis'].astype(int)
    df2['dataset'] = 1
    print('Previous Dataset:')
    print(df2.groupby('diagnosis').count())
    
    #IEEE
    df3=pd.read_csv(
            'IEEE/label/train.csv',header=1,
            names =['id', 'diagnosis'], usecols=[0,1],
            dtype={'id':str, 'diagnosis':np.int8})
    df3=df3.append(pd.read_csv(
            'IEEE/label/test.csv',header=1,
            names =['id', 'diagnosis'], usecols=[0,1],
            dtype={'id':str, 'diagnosis':np.int8}))
    df3['dataset'] =2
    print('IEEE')
    print(df3.groupby('diagnosis').count())
    df=pd.DataFrame().append(df1).append(df2).append(df3)
    print('Overall train:')
    print(df.groupby('diagnosis').count())
    print('Overall val:')
    print(df_val.groupby('diagnosis').count())
    
    
    data={'train':df, 'val':df_val}
    dataset={x: APTOSDataset(x, data[x], transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: DataLoader(dataset[x],
            batch_size=args.batch, shuffle = (x=='train'),
            num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    #messidor
    '''
    df4=pd.DataFrame()
    for i in range(1,4):
        for j in range(1,5):
            df4=df4.append(pd.read_excel(
                    'messidor/Annotation_Base'+str(i)+str(j)+'.xls',header=1,
                    names =['id', 'diagnosis'], usecols=[0,2], 
                    dtype={'id':str, 'diagnosis':np.int8}))
    df4['dataset'] = 3
    print('Messidor:')
    print(df4.groupby('diagnosis').count())
    '''
    
    print('Overall val:')
    print(df_val.groupby('diagnosis').count())
    
    
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
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                nb +=1
                num += batch
                loss = loss.item() 
                running_loss += loss * inputs.size(0)
                #propose=outputs.round().long().clamp(0,4)
                max, propose = outputs.data.max(1)
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


