# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:49:10 2019

@author: yczhang
"""
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import numpy
import cv2
import matplotlib.pyplot as plt

mean=[0.4402, 0.2334, 0.0674]
std=[0.2392, 0.1326, 0.0470]
size=224
transform= { 
 'train':transforms.Compose([
     #transforms.RandomResizedCrop(size,scale=(0.2, 1.0), ratio=(0.9, 1.11111)),
     #transforms.ColorJitter(0.3,0.1,0.1,0.04),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip()
     ]),      
 'val':transforms.Compose([
     transforms.Resize((size,size))
     ]),
'test':transforms.Compose([
     transforms.Resize((size,size))
     ])}

totensor=transforms.Compose([     
    transforms.ToTensor(),
    transforms.Normalize(mean,std)])


def crop_image(im, size=512):
    #output = image.copy()
    pil_image = im.convert('RGB') 
    open_cv_image = numpy.array(pil_image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no contours!')
        flag = 0
        return im, flag
    cnt = max(contours, key=cv2.contourArea)
    ((x, y), r) = cv2.minEnclosingCircle(cnt)
    x = int(x); y = int(y); r = int(r)
    flag = 1
    if r > 100:
        return im.crop((x-r,y-r,x+r,y+r)).resize((size,size), resample=Image.BILINEAR), (x,y,r)
   


output='train512'
if not os.path.exists(output):
    os.mkdir(output)

dirs=['exter-resized/resized_train_cropped/', 'train_image']
f=open('nocicle.txt','w')
cnt = 0
for folder in dirs:
    for f in os.listdir(folder):
        if f.endswith('png') or f.endswith('jpeg'):
            im=Image.open(os.path.join(folder,f))
            w,h = im.size
            a= np.sqrt(w*h)
            tf = transforms.Compose([
                    transforms.RandomRotation(12),
                    transforms.CenterCrop((a,a))])
            
            im, flag=crop_image(im, 512)
            if flag:
                im=transform['train'](im)
                name,_ = f.split('.')
                im.save(os.path.join(output,name+'.jpg'))
                print(name,':',flag)
                cnt+=1
            else:
                print(name,':no circle found')
                f.write(name+'\n')

f.close()
print(cnt)