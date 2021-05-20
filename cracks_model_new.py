import numpy as np
import pandas as pd
import os

for dirname,_,filenames in os.walk('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/'):
    for filename in filenames:
        print(os.path.join(dirname,filename))
        

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(cv2.imread('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/cracks/0001-2.jpg'))
plt.imshow(cv2.imread('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/cracks/0001-3.jpg'))

plt.imshow(cv2.imread('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/normal/1.jpg'))
plt.imshow(cv2.imread('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/normal/3.jpg'))

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

training_set = train_datagen.flow_from_directory('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/',
                                                 target_size = (224,224),
                                                 batch_size=32,
                                                 class_mode='binary',subset='training')

validation_generator = train_datagen.flow_from_directory('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/',
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='binary',
                                                         subset='validation')

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[224,224,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

cnn.fit(x=training_set,validation_data=validation_generator,epochs=25)
cnn.save('/home/smohammad/Projects/Raj_Project/Cracks_New/model/cracks_model24.h5')
#Accuracy - 95
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/cracks/0005-1.jpg',target_size=(224,224))
test_image = image.load_img('/home/smohammad/Projects/Raj_Project/Cracks_New/Images/normal/34.jpg',target_size=(224,224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='normal'
else:
    prediction='crack'
print(prediction)
