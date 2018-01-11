#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 09:26:30 2018

@author: nic
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 13:34:29 2017

@author: NIU
"""
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from PIL import Image
import glob
import numpy as np
import cv2
#npy_path="/workspace/tissue2/npy_path"
import matplotlib.pyplot as plt





def load_test_data(imgs_list):
    print('-' * 30)
    print('load test images...')
    print('-' * 30)
    imgs_test = np.load(imgs_list)
    #imgs_test = np.load(imgs_list[0])
    imgs_test = imgs_test.astype('float32')
    imgs_test /= 255
    #mean = imgs_test.mean(axis=0)
    mean2 = imgs_test.mean()
    imgs_test -= mean2
    return imgs_test

def creat_small_patches(imgs_test,row_assign,col_assign):
    print('-' * 30)
    print('creat_small_patches...')
    print('-' * 30)
    #row_num=int(imgs_test.shape[0]/512)
    #col_num=int(imgs_test.shape[1]/512)
    #row_pixel_not_padding=imgs_test.shape[0]%512
    #col_pixel_not_padding=imgs_test.shape[1]%512
    #row_assign=512
    #col_assign=512
    row_num=int(imgs_test.shape[0]/row_assign)
    col_num=int(imgs_test.shape[1]/col_assign)
    #row_pixel_not_padding=imgs_test.shape[0]%row_assign
    #col_pixel_not_padding=imgs_test.shape[1]%col_assign
    #imgdatas = np.ndarray(((row_num+1)*(col_num+1), imgs_test.shape[0], imgs_test.shape[1], 1), dtype=np.float32)
    imgdatas = np.ndarray(((row_num)*(col_num), row_assign, col_assign, 1), dtype=np.float32)
    count=0
    for i in range(row_num):
        for j in range(col_num):
            imgdatas[count]=imgs_test[row_assign*i:row_assign*i+row_assign,col_assign*j:col_assign*j+col_assign]
            count=count+1
            #plt.imshow(imgdatas[10].reshape(512,512),cmap='gray')
    if count==(row_num)*(col_num):
        print('count is good')
    else:
        print("count is wrong")
    imgs_test_patches=imgdatas
    return imgs_test_patches

def rebuild_img(imgs_test_predict_patches,row_assign,col_assign,imgs_test,channel):
    print('-' * 30)
    print('rebuild_img...')
    print('-' * 30)
    row_num=int(imgs_test.shape[0]/row_assign)
    col_num=int(imgs_test.shape[1]/col_assign)
    imgdatas = imgs_test_predict_patches
    #imgs_test_predict=np.zeros(imgs_test.shape)
    imgs_test_predict=np.zeros((row_num*row_assign,col_num*col_assign,channel),dtype=np.float32)
    count=0
    for i in range(row_num):
        for j in range(col_num):
            imgs_test_predict[row_assign*i:row_assign*i+row_assign,col_assign*j:col_assign*j+col_assign,:]=imgdatas[count]
            count=count+1
            #plt.imshow(imgdatas[10].reshape(512,512),cmap='gray')
            #plt.figure
            #plt.imshow(imgs_test.reshape(imgs_test.shape[0],imgs_test.shape[1]))
            #plt.figure
            #plt.imshow(imgs_test_predict.reshape(imgs_test_predict.shape[0],imgs_test_predict.shape[1]))
    return imgs_test_predict
            
#########################################
#test rebulid
'''
A=np.array([[1,1,2,2,3,3],[1,1,2,2,3,3],[4,4,5,5,6,6],[4,4,5,5,6,6,]])
A = A.reshape(A.shape+(1,))
A.shape           
plt.imshow(A)
temp=np.zeros((6,2,2,1),dtype=np.float32)
for i in range(2):
    for j in range(3):
        temp[i+j]=A[2*i:2*i+2,2*j:2*j+2]
        print(i)
        print(j)            

B=np.zeros(A.shape)
for i in range(2):
    for j in range(3):
        B[2*i:2*i+2,2*j:2*j+2]=temp[i+j]    

plt.imshow(A.reshape(4,6))         
plt.imshow(B.reshape(4,6))         
            
    return imgs_test
'''
################





def get_unet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 1))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)


    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print('model compile')
    return model

large_npy_path="/home/nic/Mycode/Unet/tissue3/large_npy_data"
large_npy_path_predict="/home/nic/Mycode/Unet/tissue3/large_npy_data_predict"
img_type="npy"


print('-' * 30)
print('Reading test npydata...')
print('-' * 30)
imgs_list = glob.glob(large_npy_path + "/*." + img_type)           # deform/train
imgs_list=sorted(imgs_list)


print ('predict test data')
for i in range(len(imgs_list)):
    imgs_test=load_test_data(imgs_list[i])
    #tempdata=np.load(imgs_list[i])
    #img_rows=tempdata.shape[0]
    #img_cols=tempdata.shape[1]
    #imgs_test_reshape=imgs_test.reshape(1,img_rows, img_cols,1)
    #img = img.reshape((1,) + img.shape)
    #imgs_test_reshape=imgs_test.reshape((1,) + imgs_test.shape)
    img_rows=512
    img_cols=512
    imgs_test_patches=creat_small_patches(imgs_test,512,512)        
    model = get_unet(img_rows, img_cols)
    #model = get_unet(512, 512)
    #model = get_unet(, )
    model.load_weights('./unet.hdf5')
    
    imgs_mask_test_predict = model.predict(imgs_test_patches, batch_size=1, verbose=1)
    #imgs_mask_test_predict = model.predict(imgs_test, batch_size=1, verbose=1)
    imgs_test_predict=rebuild_img(imgs_mask_test_predict,img_rows,img_cols,imgs_test,3)        
    imgname=imgs_list[i]
    img_name=imgname[imgname.rindex("/") + 1:imgname.rindex("." + img_type)] # 获得文件名(不包含后缀)
    np.save(large_npy_path_predict+'/'+img_name+'_predict.npy',  imgs_test_predict)
    #a=np.load(large_npy_path_predict+'/'+img_name+'_predict.npy')
    #plt.imshow(a)
    imgs_test_predict_index=imgs_test_predict.argmax(axis=2)
    imgs_test_predict_index=imgs_test_predict_index.astype('float32')
    imgs_test_predict_index[imgs_test_predict_index==0]=0
    imgs_test_predict_index[imgs_test_predict_index==1]=128
    imgs_test_predict_index[imgs_test_predict_index==2]=255
    np.save(large_npy_path_predict+'/'+img_name+'_predict_convert.npy',  imgs_test_predict_index)
   # plt.figure(3),plt.imshow(imgs_test_predict_index)
    print(i)