#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 21:59:12 2017

@author: nic
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
#from PIL import Image
import glob
import numpy as np

class mygeneratedata(object):
    """
    A class used to augmentate image
    Firstly, read train image and label seperately, and then merge them together for the next process
    Secondly, use keras preprocessing to augmentate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, out_rows,out_cols,train_path="/home/nic/Mycode/Unet/tissue2/data/train/image_gray", 
                 label_path="/home/nic/Mycode/Unet/tissue2/data/train/label_gray", 
                 test_path="/home/nic/Mycode/Unet/tissue2/data/test/image_gray",
                 test_label_path="/home/nic/Mycode/Unet/tissue2/data/test/label_gray",
                 npy_path="/home/nic/Mycode/Unet/tissue2/npydata",
                 img_type="jpg"):

        """
        Using glob to get all .img_type form path
        """
        self.out_rows = out_rows
        self.out_cols = out_cols
        #self.train_imgs = glob.glob(train_path + "/*." + img_type)  # 训练集
        #self.label_imgs = glob.glob(label_path + "/*." + img_type)  # label
        self.train_path = train_path
        self.label_path = label_path
        self.test_path = test_path
        self.test_label_path = test_label_path
        self.npy_path=npy_path        
        self.img_type = img_type
 
        
    def create_train_data(self):
        # 将增强之后的训练集生成npy         

        print('-' * 30)
        print('creating train image')
        print('-' * 30)
        trainPath=self.train_path
        labelPath=self.label_path
        imgs = glob.glob(trainPath + '/*' + '.jpg')
        imgs=sorted(imgs)
        labels = glob.glob(labelPath + '/*' + '.jpg')
        labels=sorted(labels)
        
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(labels), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for i in range(len(imgs)):
            
            #img = plt.imread(imgs[i])
            #label = plt.imread(labels[i])
            img = load_img(imgs[i], grayscale=True)
            label = load_img(labels[i], grayscale=True) 
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
                #if i % 100 == 0:
                    #print('Done: {0}/{1} images'.format(i, len(imgs)))
            print(i)
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')
        
#################################################################################################################        
       

    def create_test_data(self):
        # 测试集生成npy
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs1 = glob.glob(self.test_path + "/*." + self.img_type)           # deform/train
        imgs1=sorted(imgs1)
        labels1 = glob.glob(self.test_label_path + '/*' + '.jpg')
        labels1=sorted(labels1)
        
        
        imgdatas1 = np.ndarray((len(imgs1), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels1 = np.ndarray((len(labels1), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        
        for i in range(len(labels1)):
            #img = plt.imread(imgs[i])
            #label = plt.imread(labels[i])
            
            label = load_img(labels1[i], grayscale=True)
            img = load_img(imgs1[i], grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas1[i] = img
            imglabels1[i] = label
            print(i)

        print('loading done', imgdatas1.shape)
        np.save(self.npy_path + '/imgs_test.npy', imgdatas1)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/imgs_mask_test.npy', imglabels1)
        print('Saving to .npy files done.')



    def load_train_data(self):
        # 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.66] = 1
        imgs_mask_train[imgs_mask_train <= 0.33] = 0
        imgs_mask_train[(imgs_mask_train >0.33)&(imgs_mask_train <= 0.66)] = 0.4
        """        
        a=np.array([[0.1, 0.5, 0.9],[0.4, 0.6 ,0.8]])
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if (a[i,j]>=0.33)&(a[i,j]<0.66):
                    a[i,j]=0.4
        """                    
        #return imgs_train, imgs_mask_train
        print('process train data')
        np.save(self.npy_path + '/imgs_train_p.npy', imgs_train)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/imgs_mask_train_p.npy', imgs_mask_train)
        print('process train data .npy files done.')
        return imgs_train, imgs_mask_train

        
    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        #return imgs_test
        print('process test data')
        np.save(self.npy_path + '/imgs_test_p.npy', imgs_test)            # 将30张训练集和30张label生成npy数据
        #np.save(self.npy_path + '/imgs_mask_test_p.npy', imgs_mask_train)
        print('process test data .npy files done.')

if __name__ == "__main__":
    aug = mygeneratedata(512,512)
    aug.create_train_data()
    aug.create_test_data()
    aug.load_train_data()
    aug.load_test_data()
    imgs_train, imgs_mask_train = aug.load_train_data()
    print(imgs_train.shape, imgs_mask_train.shape)