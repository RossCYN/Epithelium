#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from data import dataProcess

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))





class dataProcess(object):
    def __init__(self, out_rows, out_cols, npy_path="/workspace/tissue2/npy_path"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.npy_path = npy_path

#############################################################################################################################


    def load_train_data(self):
        # 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        #imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        a = np.load(self.npy_path + "/imgs_mask_train.npy")
        a = np.load("npydata/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        #imgs_mask_train = imgs_mask_train.astype('float32')
        a=a.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        #imgs_mask_train /= 255
        #a/=255
       
        b=np.zeros([len(a),a.shape[1],a.shape[2],3],dtype='float32')
        #imgs_mask_train[imgs_mask_train > 0.66] = 2
        #imgs_mask_train[imgs_mask_train <= 0.33] = 0
        #imgs_mask_train[(imgs_mask_train > 0.33)&(imgs_mask_train <= 0.66)] = 1
        
        index_temp0=(a <= 85)
        index_0=index_temp0.repeat(3,axis=3)
        index_0[:,:,:,1]=False
        index_0[:,:,:,2]=False
        b[index_0]=1
        
        index_temp1=(a>85)&(a <= 170)
        index_1=index_temp1.repeat(3,axis=3)
        index_1[:,:,:,0]=False
        index_1[:,:,:,2]=False
        b[index_1]=1
        
        index_temp2=(a > 170)
        index_2=index_temp2.repeat(3,axis=3)
        index_2[:,:,:,0]=False
        index_2[:,:,:,1]=False
        b[index_2]=1 
        
        imgs_mask_train=b
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
        return imgs_test



class myUnet(object):
    def __init__(self, npy_path="/workspace/tissue2/npy_path",img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.npy_path = npy_path
        self.img_cols = img_cols
		

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

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

        model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy']) #loss='binary_crossentropy',loss='categorical_crossentropy'
        print('model compile')
        return model

    

    def train(self):
        print "loading data"
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print "loading data done"
        model = self.get_unet()
        print "got unet"

        # 保存的是模型和权重,
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print 'Fitting model...'
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=20, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

        print 'predict test data'
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)

    def test(self):
        print "loading data"
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print "loading data done"
        model = self.get_unet()
        print "got unet"
        model.load_weights('./unet.hdf5')
        print 'predict test data'
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('imgs_mask_test.npy', imgs_mask_test)
	

if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()








