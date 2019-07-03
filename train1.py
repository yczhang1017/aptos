
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

#import torch.nn.functional as F

#from torch.autograd import Variable

#import torch.utils.model_zoo as model_zoo



parser = argparse.ArgumentParser(
    description='train a model')
parser.add_argument('--root', default='./',
                    type=str, help='directory of the data')
parser.add_argument('--batch_size', default=24, type=int,
                    help='Batch size for training')
parser.add_argument('--workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('-e','--epochs', default=35, type=int,
                    help='number of epochs to train')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--model', default='nasnetamobile', type=str,
                    help='model name')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from') 
parser.add_argument('--size', default=288, type=int,
                    help='image size')


args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

mean=[0.4402, 0.2334, 0.0674]
std=[0.2392, 0.1326, 0.0470]
transform= { 
 'train':transforms.Compose([
     transforms.Resize((args.size,args.size),interpolation=0),
     transforms.ColorJitter(),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ]),      
 'val':transforms.Compose([
     transforms.Resize((args.size,args.size),interpolation=0),
     transforms.ToTensor(),
     transforms.Normalize(mean,std)
     ])}

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
    
class APTOSDataset(torch.utils.data.Dataset):
    def __init__(self, root, phase, data ,transform):
        self.root= root
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
        
        img_name = os.path.join(self.root,
                                x + '.png')
        image = PIL.Image.open(img_name)
        w,h = image.size
        a= (w+h)//2
        tf = transforms.Compose([
                transforms.RandomRotation(25),
                transforms.CenterCrop((a,a))])
        image = self.transform(tf(image))
        if self.phase in ['train','val']:
            return image, y
        elif self.phase == 'test' :
            return image 
        
        
def main():
    train_csv=os.path.join(args.root, 'train.csv')
    df  = pd.read_csv(train_csv)
    data={'train':None,'val':None}
    dataset={'train':None,'val':None}
    dataloader={'train':None,'val':None}
    data['train'], data['val'] = \
        train_test_split(df.values.tolist(), test_size=0.1, random_state=42)  
    image_folder = os.path.join(args.root,'train_image')
    dataset={x: APTOSDataset(image_folder, x, data[x], transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,
            num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    
    model = pretrainedmodels.__dict__[args.model](num_classes=1000, pretrained='imagenet')
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential( 
            nn.BatchNorm1d(1056, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=1056, out_features=512, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1, bias=True),
            )
    #print(model)
    model = model.to(device)
    if torch.cuda.is_available():
        model=nn.DataParallel(model)
        cudnn.benchmark = True    
    if args.checkpoint:
        print('Resuming training from epoch {}, loading {}...'
              .format(args.resume_epoch,args.checkpoint))
        weight_file=os.path.join(args.root,args.checkpoint)
        model.load_state_dict(torch.load(weight_file,
                                 map_location=lambda storage, loc: storage))    

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=args.lr, 
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[20,25,30], gamma=0.1)
   
    for i in range(args.resume_epoch):
        scheduler.step()
    for epoch in range(args.resume_epoch,args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            num = 0
            running_loss = 0
            for inputs,targets in dataloader[phase]:
                t1 = time.time()
                nb = inputs.size(0)
                inputs = inputs.to(device)                
                targets= targets.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.reshape(nb)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                num += nb
                loss = loss.item() 
                running_loss += loss * inputs.size(0)
                propose=torch.round(outputs)
                acc = (propose==targets)/inputs.size(0)
                t2 = time.time()
                if num % (1)==0:
                    print(propose)
                    print(targets)
                    print('l: {:.4f} | {:.4f}, p: {:.4f} r, t:{:.4f}' \
                          .format(loss, running_loss, acc, t2-t1))
            if phase == 'val':
                torch.save(model.state_dict(),
                          os.path.join(args.save_folder,'out_'+str(epoch+1)+'.pth'))
        print()
        
if __name__ == '__main__':
    main()        


