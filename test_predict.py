import matplotlib.pyplot as plt
import numpy as np
import glob
#from PyQt4 import QtCore, QtGui







# mydata = dataProcess(512,512)

imgs_test = np.load('/home/nic/Mycode/Unet/tissue2/npydata/imgs_test.npy')

test_label_gray_path="/home/nic/Mycode/Unet/tissue2/data/test/label_gray"
imgs_test_true = glob.glob(test_label_gray_path + "/*." + 'tif')
# myunet = myUnet()
#
# model = myunet.get_unet()
#
# model.load_weights('unet.hdf5')
#
# imgs_mask_test = model.predict(imgs_test, verbose=1)
#
# np.save('imgs_mask_test.npy', imgs_mask_test)
#imgs_label_train = np.load('/home/nic/Mycode/Unet/tissue/npydata/imgs_mask_train.npy')
imgs_train=np.load('/home/nic/Mycode/Unet/tissue2/npydata/imgs_train.npy')

imgs_test_predict = np.load('/home/nic/Mycode/Unet/tissue2/imgs_mask_test.npy')

imgs_test_predict_index = np.load('/home/nic/Mycode/Unet/tissue2/imgs_test_predict_index.npy')


imgs_test_predict_origin = np.load('/home/nic/Mycode/Unet/tissue2/npydata/imgs_mask_test.npy')



print(imgs_test.shape, imgs_test_predict.shape)



'''
n = 2
plt.figure(figsize=(20, 4))
for i in range(20, 22):
    plt.gray()
    ax = plt.subplot(2, n, (i-20)+1)
    plt.imshow(imgs_test[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, (i - 20) + n + 1)
    plt.imshow(imgs_test_predict[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

a=imgs_test_predict[10].reshape(512,512,3)
plt.imshow(a)

b=imgs_test_predict_origin[10].reshape(512,512)
plt.imshow(b)

b=imgs_test_predict_origin[0].reshape(512,512)
plt.imshow(b)

L=10
for i in range(L):
    plt.subplot(4,L,i+1)
    plt.imshow(imgs_test[10+i].reshape(512,512),cmap='gray')
        
    plt.subplot(4,L,L+i+1)
    plt.imshow(imgs_test_predict_origin[10+i].reshape(512,512))
    
    plt.subplot(4,L,2*L+i+1)
    plt.imshow(imgs_test_predict_index[10+i].reshape(512,512))    

    plt.subplot(4,L,3*L+i+1)
    plt.imshow(imgs_test_predict[10+i].reshape(512,512,3))




%matplotlib qt
plt.figure()
for i in range(len(label_imgs)):
    plt.subplot(3,len(label_imgs),i+1)
    plt.imshow(imgs_test[i].reshape(512,512))
    
    plt.subplot(3,len(label_imgs),len(label_imgs)+i+1)
    plt.imshow(plt.imread(imgs_test_true[i]))
    
    
    plt.subplot(3,len(label_imgs),2*len(label_imgs)+i+1)
    plt.imshow(imgs_test_predict[i].reshape(512,512))
    plt.show()
    
plt.show()
