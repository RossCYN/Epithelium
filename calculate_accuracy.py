#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:22:20 2018

@author: nic
"""
import numpy as np


imgs_test_predict = np.load('/home/nic/Mycode/Unet/tissue2/imgs_mask_test.npy')
K=imgs_test_predict[0,:,:,:]

a=imgs_test_predict[0:5,:,:,:]
m=np.max(a,axis=3)
c=a.argmax(axis=3)


imgs_test_mask_true = np.load('/home/nic/Mycode/Unet/tissue2/npydata/imgs_mask_test.npy')
d=imgs_test_mask_true[10,:,:,:] 

imgs_test_mask_true=imgs_test_mask_true.astype('float32')
imgs_test_mask_true[imgs_test_mask_true<=85]=0
imgs_test_mask_true[(imgs_test_mask_true>85)&(imgs_test_mask_true <=150)]=128
imgs_test_mask_true[imgs_test_mask_true>150]=255

a=imgs_test_mask_true.reshape(309,512,512)
e=imgs_test_mask_true[10,:,:,:] 
d==e

imgs_test_predict_index=imgs_test_predict.argmax(axis=3)
imgs_test_predict_index=imgs_test_predict_index.astype('float32')
imgs_test_predict_index[imgs_test_predict_index==0]=0
imgs_test_predict_index[imgs_test_predict_index==1]=128
imgs_test_predict_index[imgs_test_predict_index==2]=255

np.save('imgs_test_predict_index.npy',imgs_test_predict_index)
b=imgs_test_predict_index.reshape(imgs_test_predict_index.shape+(1,))
#num=(imgs_test_predict_index==imgs_test_mask_true)
num=(imgs_test_predict_index==a)

accurate=0
for i in range(num.shape[0]):
    for j in range(num.shape[1]):
        for k in range(num.shape[2]):
            if num[i,j,k]==True:
                accurate=accurate+1
print(accurate/(num.shape[0]*num.shape[1]*num.shape[2]))
            