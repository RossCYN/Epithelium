#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:04:56 2018

@author: nic
"""

import numpy as np
#from keras.models import *
#from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization
#from keras.optimizers import *
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from PIL import Image
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt


import matplotlib.patches as mpatches
from skimage import data,filters,segmentation,measure,morphology,color


img_original_path="/home/nic/Mycode/Unet/tissue3/data/original_jpg"
img_type="jpg"
imgs_list = glob.glob(img_original_path + "/*." + img_type)           # deform/train
imgs_list=sorted(imgs_list)



#img_original_path="/home/nic/Mycode/Unet/tissue3/data/original"
true_label="/home/nic/Mycode/Unet/tissue3/data/original_label"

final_index="/home/nic/Mycode/Unet/tissue3/final_index"

index_type="npy"
#index_list = glob.glob(final_index + "/*without_closed." + index_type)           # deform/train
index_list = glob.glob(final_index + "/*." + index_type)
index_list=sorted(index_list)

final_epithelium="/home/nic/Mycode/Unet/tissue3/final_epithelium"

index_closed_list=[]
j=0
for i in range(0,len(index_list),2):
    index_closed_list.append(index_list[i])
    j=j+1

#index_closed_list[0]='0'
i=0
for imgname in index_closed_list:
    imgname[imgname.rindex('/')+1:imgname.rindex('.npy')]
    i=i+1
    print('process img    ',i)
    print(imgname[imgname.rindex('/')+1:imgname.rindex('_closed.npy')])
    for img_original_name in imgs_list:
        if imgname[imgname.rindex('/')+1:imgname.rindex('_closed.npy')]==img_original_name[img_original_name.rindex('/')+1:img_original_name.rindex('.jpg')]:
            print('name matched')
            plt.figure(i)
            plt.subplot(1,2,1)
            epithelium=np.load(imgname)
            Original_img=plt.imread(img_original_name)
            #plt.imshow(epithelium,cmap='gray')
            Original_img_crop=Original_img[0:epithelium.shape[0],0:epithelium.shape[1]]
            plt.imshow(Original_img_crop)
            
            plt.subplot(1,2,2)
            #epithelium=np.load(imgname)
            #plt.imshow(epithelium,cmap='gray')
            #Original_img_crop=Original_img[0:epithelium.shape[0],0:epithelium.shape[1]]
            #Img_temp=np.zeros(Original_img_crop.shape)
            Img_temp=np.ones(Original_img_crop.shape)
            #plt.imshow(Img_temp)
            # if not uint8 color will change
            Img_temp_=Img_temp.astype('uint8')*255
            #plt.imshow(Img_temp_)
            #Img_temp[epithelium==1]=Original_img_crop[epithelium==1]
            Img_temp_[epithelium==1]=Original_img_crop[epithelium==1]
            #plt.imshow(Img_temp)
            plt.imshow(Img_temp_)
            plt.savefig(final_epithelium+'/'+imgname[imgname.rindex('/')+1:imgname.rindex('_closed.npy')]+'.jpg')
            print('already save plt',i)
            print('=='*40)
            print('')
print('Done')
            
            
            
            
            #plt.imshow(np.load(imgname))
    
    