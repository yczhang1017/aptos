# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:49:10 2019

@author: yczhang
"""
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import cv2
import torch

transform=transforms.Compose([
        transforms.Resize(1024)
     ]) 
totensor=transforms.Compose([     
    transforms.ToTensor()])


def circle(im):
    #output = image.copy()
    pil_image = im.convert('RGB') 
    open_cv_image = np.array(pil_image)
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
        return (x,y,r)    

dirs=['train_image']
outputs=['train1024']

cnt = 0
mean= torch.zeros((3))
std= torch.zeros((3))

for folder, output in zip(dirs,outputs):
    if not os.path.exists(output):
        os.mkdir(output)
    for f in os.listdir(folder):
        if f.endswith('png') or f.endswith('jpeg'):
            name,_ = f.split('.')
            im=Image.open(os.path.join(folder,f))
            w,h = im.size
            flag = circle(im)
            if flag:
                x,y,r=flag
                x1= max(x-r,0)
                y1= max(y-r,0)
                x2= min(x+r,w)
                y2= min(y+r,h)
                if x1>0 or y1>0 or x2<w or y2<h:
                    print(f,':',flag,(w,h),(x1,x2,y1,y2))
                    im=im.crop((x1,y1,x2,y2))
            cnt+=1
            im=transform(im)
            im.save(os.path.join(output,name+'.jpg'))
            tensor = totensor(im)
            mean += tensor.mean(dim=(1,2)) 
            std += tensor.std(dim=(1,2))


print(cnt)
print(mean/cnt)
print(std/cnt)

