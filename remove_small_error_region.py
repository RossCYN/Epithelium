#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:16:03 2018

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





#true_index=img_true_label_temp
#imgs_final_predict_index=img_temp_remove_small            
def calculate_accuracy_precision_recal(imgs_final_predict_index,true_index):
    print('-' * 30)
    print('calculate_accuracy_precision_recal...')
    print('-' * 30)
    img_label_temp=np.zeros(true_index.shape)
    img_label_temp[true_index==128]=1
    plt.imshow(img_label_temp)
    plt.imshow(imgs_final_predict_index)
    
    img_label_temp_crop=img_label_temp[0:imgs_final_predict_index.shape[0],0:imgs_final_predict_index.shape[1]]
    A_num=(img_label_temp_crop==imgs_final_predict_index)
#    they_both_have=0
#    
#    for i in range(A_num.shape[0]):
#        for j in range(A_num.shape[1]):
#                if A_num[i,j]==True:
#                    they_both_have=they_both_have+1
    
    #  c+d=they_both_have
    #Temp_num=np.zeros(img_label_temp_crop.shape)
    #Temp_num[A_num==1]=1
    they_both_have=sum(sum(A_num))
    #they_both_have=sum(sum(Temp_num))
#    print(accurate/(num.shape[0]*num.shape[1]*num.shape[2]))
    #A
    all_fianl_predict_positive=sum(sum(imgs_final_predict_index))
    #B
    all_true_positive=sum(sum(img_label_temp_crop))
    #All
    All=imgs_final_predict_index.shape[0]*imgs_final_predict_index.shape[1]
    #A+B-c+d=All,c+d=they_both_have
    c_=(all_fianl_predict_positive+all_true_positive+they_both_have-All)/2
    precision=c_/all_fianl_predict_positive
    recall=c_/all_true_positive
    print('precision is ',precision)
    print('recall is ',recall)
    return precision,recall
        



img_original_path="/home/nic/Mycode/Unet/tissue3/data/original_jpg"

#img_original_path="/home/nic/Mycode/Unet/tissue3/data/original"
true_label="/home/nic/Mycode/Unet/tissue3/data/original_label"

#img_original_type="tif"
img_original_type="jpg"
imgs_original_list = glob.glob(img_original_path + "/*." + img_original_type)           # deform/train
imgs_original_list=sorted(imgs_original_list)

img_label_type="tif"
imgs_label_list = glob.glob(true_label + "/*." + img_label_type)           # deform/train
imgs_label_list=sorted(imgs_label_list)



large_npy_path_predict="/home/nic/Mycode/Unet/tissue3/large_npy_data_predict_convert"
plot_path="/home/nic/Mycode/Unet/tissue3/final_result"
final_index="/home/nic/Mycode/Unet/tissue3/final_index"
img_type="npy"
imgs_list = glob.glob(large_npy_path_predict + "/*." + img_type)           # deform/train
imgs_list=sorted(imgs_list)


Precision=np.zeros(len(imgs_list))
Recall=np.zeros(len(imgs_list))
Precision_closed=np.zeros(len(imgs_list))
Recall_closed=np.zeros(len(imgs_list))

for i in range(11,len(imgs_list)):
    print('load image    ', i)
    imgs_test_predict_index=np.load(imgs_list[i])
    #plt.figure(i)
    #plt.imshow(imgs_test_predict_index)
    #imgs_test_predict=img_predict_original
    #imgs_test_predict_index=imgs_test_predict.argmax(axis=2)
    #imgs_test_predict_index=imgs_test_predict_index.astype('float32')

# =============================================================================
#      imgs_test_predict_index[imgs_test_predict_index==0]=0
#      imgs_test_predict_index[imgs_test_predict_index==1]=128
#      imgs_test_predict_index[imgs_test_predict_index==2]=255
#      np.save(large_npy_path_predict+'/'+img_name+'_predict_convert.npy',  imgs_test_predict_index)
# =============================================================================

    #plt.imshow(imgs_test_predict_index)
    #plt.imshow(imgs_test_predict_index,cmap='gray')

    #img=imgs_test_predict_index[:]
    #img.shape
    img=imgs_test_predict_index
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(50, 50))  

##闭运算  
#closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  
##显示腐蚀后的图像  
#cv2.imshow("Close",closed) 
#plt.imshow(closed) 
#plt.figure(2)
#plt.imshow(imgs_test_predict_index) 
#开运算  
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  
    print('open process done')
    #显示腐蚀后的图像  
    #cv2.imshow("Open", opened);  
    #plt.imshow(opened) 
    #plt.imshow(img)
    img_temp=np.zeros(imgs_test_predict_index.shape,dtype='float32') 
    #img_temp2=np.zeros(imgs_test_predict_index.shape,dtype='float32')
    img_temp3=np.zeros(imgs_test_predict_index.shape,dtype='float32')
    # temp  class 1 and background
    img_temp[imgs_test_predict_index==128]=1       

    img_temp3[opened==128]=1
    #plt.imshow(img_temp3)



# =============================================================================
#     a=measure.label(img_temp3, connectivity=None)
#     plt.imshow(a,cmap='gray')
#     plt.figure(100)
#     plt.imshow(a,cmap='gray')
# =============================================================================
   #closed
    c=morphology.remove_small_objects(img_temp3==1, min_size=600*600, connectivity=1, in_place=False)
    
    img_temp3_remove_small=np.zeros(imgs_test_predict_index.shape,dtype='float32')
    img_temp3_remove_small[c]=1
    #plt.imshow(img_temp4)
   
    # with out closed 
    d=morphology.remove_small_objects(img_temp==1, min_size=600*600, connectivity=1, in_place=False)
    img_temp_remove_small=np.zeros(imgs_test_predict_index.shape,dtype='float32')
    img_temp_remove_small[d]=1
# =============================================================================
#     plt.subplot(2,3,1)
#     plt.imshow(imgs_test_predict_index)
#     plt.subplot(2,3,2)
#     plt.imshow(img_temp)
#     plt.subplot(2,3,3)
#     plt.imshow(img_temp_remove_small)
#     plt.subplot(2,3,4)
#     plt.imshow(opened)
#     plt.subplot(2,3,5)
#     plt.imshow(img_temp3)
#     plt.subplot(2,3,6)
#     plt.imshow(img_temp3_remove_small)
# =============================================================================
    
    imgname=imgs_list[i]
    plt.figure(i)
    plt.subplot(2,5,1)
    for j in range(len(imgs_original_list)):
        imgs_original_name=imgs_original_list[j]
        if imgs_original_name[imgs_original_name.rindex("/") +1:imgs_original_name.rindex(".jpg")]==imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]:
            print('match the original image',j)
            plt.imshow(plt.imread(imgs_original_list[j]))
            plt.title('Original image')
    # original predict result
    plt.subplot(2,5,2)        
    plt.imshow(imgs_test_predict_index)
    plt.title('Model predict')
    # closed
    plt.subplot(2,5,3)
    plt.imshow(img_temp)
    
    # get rid of small patches
    plt.subplot(2,5,4)
    plt.imshow(img_temp_remove_small)
    plt.title('Remove without closed')

    # true label
    plt.subplot(2,5,5)
    for k in range(len(imgs_label_list)):
        imgs_label_name=imgs_label_list[k]
        if imgs_label_name[imgs_label_name.rindex("/") +1:imgs_label_name.rindex("_flag.tif")]==imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]:
            img_true_label_temp=cv2.imread(imgs_label_list[k],cv2.IMREAD_GRAYSCALE)
            print('match the label image',k)
            plt.imshow(cv2.imread(imgs_label_list[k],cv2.IMREAD_GRAYSCALE))
            plt.title('True label')
    
  
    precision_without_closed,recall_without_closed=calculate_accuracy_precision_recal(img_temp_remove_small,img_true_label_temp)    
    precision_with_closed,recall_with_closed=calculate_accuracy_precision_recal(img_temp3_remove_small,img_true_label_temp) 
    
    Precision[i]=precision_without_closed
    Recall[i]=recall_without_closed
    Precision_closed[i]=precision_with_closed
    Recall_closed[i]=recall_with_closed
    
    plt.subplot(2,5,6)
    for l in range(len(imgs_original_list)):
        imgs_original_name=imgs_original_list[l]
        if imgs_original_name[imgs_original_name.rindex("/") +1:imgs_original_name.rindex(".jpg")]==imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]:
            plt.imshow(Image.open(imgs_original_list[l]))
            plt.title('Original image')
    
    plt.subplot(2,5,7)
    plt.imshow(opened)
    plt.title('Closed process')
    plt.subplot(2,5,8)
    plt.imshow(img_temp3)
    
    plt.subplot(2,5,9)
    plt.imshow(img_temp3_remove_small)
    plt.subplot(2,5,10)
    for m in range(len(imgs_label_list)):
        imgs_label_name=imgs_label_list[m]
        if imgs_label_name[imgs_label_name.rindex("/") +1:imgs_label_name.rindex("_flag.tif")]==imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]:
            
            plt.imshow(cv2.imread(imgs_label_list[m],cv2.IMREAD_GRAYSCALE),cmap='gray')
            plt.title('True label')
        
    # save result
    plt.savefig(plot_path+ '/'+str(i)+'.jpg')
    
    
    plt.savefig(plot_path+ '/'+imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]+'.jpg')
    np.save(final_index+ '/'+imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]+'_closed.npy',img_temp3_remove_small)
    np.save(final_index+ '/'+imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]+'_without_closed.npy',img_temp_remove_small)
    
    print('Already produce '+str(i))
    
print('Done')    
    #imgname[imgname.rindex("/") +1:imgname.rindex("_predict_convert.npy")]
np.save(final_index+ '/'+'Precision.npy',Precision)
np.save(final_index+ '/'+'Recall.npy',Recall)
np.save(final_index+ '/'+'Precision_closed.npy',Precision_closed)    
np.save(final_index+ '/'+'Recall_closed.npy',Recall_closed)   
    
#A=np.array([[1,1,2,2,3,3],[1,1,2,2,3,3]])
#At=np.zeros(A.shape,dtype='float32')
#At1=np.zeros(A.shape,dtype='float32')
#
#At[A==2]=4
#At1[A==3]=5
    
    
#opened=img
#for i in range(5):
#    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernel2)  
#    plt.figure(i+5)
#    plt.imshow(opened)



#a=measure.label(imgs_test_predict_index)
#plt.imshow(a)

  
#cv2.waitKey(0)  
#cv2.destroyAllWindows()  

