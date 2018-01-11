#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:41:44 2018

@author: nic
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2

test_path="/home/nic/Mycode/Unet/tissue3/data/original"
large_npy_path="/home/nic/Mycode/Unet/tissue3/large_npy_data"
img_type="tif"


print('-' * 30)
print('Creating test images...')
print('-' * 30)
imgs_list = glob.glob(test_path + "/*." + img_type)           # deform/train
imgs_list=sorted(imgs_list)

for i in range(len(imgs_list)):
    imgname=imgs_list[i]
    img_name=imgname[imgname.rindex("/") + 1:imgname.rindex("." + img_type)] # 获得文件名(不包含后缀)
    img_pixel=cv2.imread(imgs_list[i], cv2.IMREAD_GRAYSCALE)
    img_pixel_array = img_to_array(img_pixel)
    out_rows=img_pixel.shape[0] 
    out_cols=img_pixel.shape[1]

        #labels1 = glob.glob(self.test_label_path + '/*' + self.img_type)
        #labels1=sorted(labels1)
        
    imgdatas = np.ndarray((out_rows, out_cols, 1), dtype=np.float32)
    imgdatas = img_pixel_array
            #imglabels1[i] = label
    print(i)
    print('loading done', imgdatas.shape)
    np.save(large_npy_path + '/'+img_name+'.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        #np.save(self.npy_path + '/imgs_mask_test.npy', imglabels1)
    print('Saving to .npy files done.')
        




