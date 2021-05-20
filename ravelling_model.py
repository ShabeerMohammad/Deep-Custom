
                    
#python36 -m pip install --upgrade tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

Ravel_imagePaths = list(paths.list_images('./Images/'))
print('Loading ' + str(len(Ravel_imagePaths )) + ' images...')

data_ravel = []
labels_ravel = []
for imagePath in Ravel_imagePaths:    
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data_ravel.append(image)
    labels_ravel.append(label)
data_ravel = np.array(data_ravel, dtype="float32")
labels_ravel = np.array(labels_ravel)

baseModel_ravel= MobileNetV2(weights="imagenet", include_top=False,input_shape=(224, 224, 3))

headModel_ravel = baseModel_ravel.output
headModel_ravel = AveragePooling2D(pool_size=(7, 7))(headModel_ravel)
headModel_ravel = Flatten(name="flatten")(headModel_ravel)
headModel_ravel = Dense(128, activation="relu")(headModel_ravel)
headModel_ravel = Dense(128, activation="relu")(headModel_ravel)
headModel_ravel = Dropout(0.3)(headModel_ravel)
headModel_ravel = Dense(2, activation="sigmoid")(headModel_ravel)

model = Model(inputs=baseModel_ravel.input, outputs=headModel_ravel)

for layer in baseModel_ravel.layers:
	layer.trainable = False

lb = LabelBinarizer()
labels_ravel = lb.fit_transform(labels_ravel)
labels_ravel = to_categorical(labels_ravel)

(trainX, testX, trainY, testY) = train_test_split(data_ravel, labels_ravel,
	test_size=0.20, stratify=labels_ravel, random_state=42)

aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

INIT_LR = 1e-4   #learning rate
EPOCHS =20       
BS = 4
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

model.save('ravelling_model.h5')

