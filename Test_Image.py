from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
from keras.callbacks import TensorBoard
import cv2
import os
import keras
from keras.models import load_model

input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())

total = 29700
traincount = int(0.7*total)
valcount = int(0.85*total)
x_train = np.zeros((traincount, 64,64,1))
x_val = np.zeros((valcount-traincount, 64,64,1))
x_test = np.zeros((total-valcount, 64,64,1))

for i in range(0, total):
    if (i>=0) and (i<traincount):
        x_train[i,:,:,0] = cv2.imread(os.path.join("SynImages/", str(i)+'.jpg'),cv2.IMREAD_GRAYSCALE)/255
    elif (i>=traincount) and (i<(valcount)):
        x_val[i-traincount,:,:,0] = cv2.imread(os.path.join("SynImages/", str(i)+'.jpg'),cv2.IMREAD_GRAYSCALE)/255
    else:
        x_test[i-valcount,:,:,0] = cv2.imread(os.path.join("SynImages/", str(i)+'.jpg'),cv2.IMREAD_GRAYSCALE)/255
        

autoencoder.load_weights("autoencoder.hdf5")

decoded_imgs = autoencoder.predict(x_test[0:1,:,:,0:1])
cv2.imshow('a', (x_test[0,:,:,0]*255).astype('uint8'))
cv2.imshow('b', (decoded_imgs[0,:,:,0]*255).astype('uint8'))
cv2.waitKey(0)
