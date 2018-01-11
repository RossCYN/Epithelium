#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:00:58 2017

@author: nic
"""

#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image

train_path="/home/nic/Mycode/Unet/tissue2/data/train/image"
label_path="/home/nic/Mycode/Unet/tissue2/data/train/label"

img_type="jpg"

train_path="/home/nic/Mycode/Unet/tissue2/data/train/image"
train_gray_path="/home/nic/Mycode/Unet/tissue2/data/train/image_gray"
test_path="/home/nic/Mycode/Unet/tissue2/data/test/image"
test_gray_path="/home/nic/Mycode/Unet/tissue2/data/test/image_gray"

test_label_path="/home/nic/Mycode/Unet/tissue2/data/test/label"
train_label_path="/home/nic/Mycode/Unet/tissue2/data/test/label"

if not os.path.lexists(train_gray_path):
    os.mkdir(train_gray_path)
if not os.path.lexists(test_gray_path):
    os.mkdir(test_gray_path)

train_label_gray_path="/home/nic/Mycode/Unet/tissue2/data/train/label_gray"
test_label_gray_path="/home/nic/Mycode/Unet/tissue2/data/test/label_gray"

if not os.path.lexists(train_label_gray_path):
    os.mkdir(train_label_gray_path)
if not os.path.lexists(test_label_gray_path):
    os.mkdir(test_label_gray_path)



train_imgs = glob.glob(train_path + "/*." + img_type)  # 训练集
#train_imgs=sorted(train_imgs, key=lambda student : student[2])
train_imgs=sorted(train_imgs)
train_label = glob.glob(label_path + "/*." + img_type)  # label
train_label=sorted(train_label)

test_imgs = glob.glob(test_path + "/*." + img_type)  # 训练集
test_imgs=sorted(test_imgs)
test_label=glob.glob(test_label_path + "/*." + img_type)
test_label=sorted(test_label)


for i in range(len(train_imgs)):
    im_gray=Image.open(train_imgs[i]).convert("L")
    im_gray.save(train_gray_path+"/"+str(i)+'.jpg')
    
for i in range(len(test_imgs)):
    im_gray=Image.open(test_imgs[i]).convert("L")
    im_gray.save(test_gray_path+"/"+str(i)+'.jpg')

for i in range(len(train_label)):
    im_gray=Image.open(train_label[i]).convert("L")
    im_gray.save(train_label_gray_path+"/"+str(i)+'.jpg')
for i in range(len(test_label)):
    im_gray=Image.open(test_label[i]).convert("L")
    im_gray.save(test_label_gray_path+"/"+str(i)+'.jpg')





    