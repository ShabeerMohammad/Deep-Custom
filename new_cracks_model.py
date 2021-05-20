
                    
#python36 -m pip install --upgrade tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2,VGG16,resnet50

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input,Activation
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

imagePaths = list(paths.list_images('./New Cracks/New_Cracks1'))
print('Loading ' + str(len(imagePaths)) + ' images...')

data = []
labels = []
for imagePath in imagePaths:    
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)
data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.3, stratify=labels, random_state=42)

baseModel = MobileNetV2(weights="imagenet", include_top=False,input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(32, activation="relu")(headModel)
#headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)


baseModel = resnet50.ResNet50(weights="imagenet", include_top=False,input_shape=(224, 224, 3))
baseModel = resnet50.ResNet50(weights="imagenet", include_top=False,input_shape=(224, 224, 3))
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
#headModel = Dropout(0.5)(headModel)
headModel = Dense(3)(headModel)
headModel = Activation("softmax")(headModel)


model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

learn_rate = 0.001
BS = 4
EPOCHS = 20
print("[INFO] compiling model...")
opt = Adam(lr=learn_rate, decay=learn_rate / EPOCHS)
opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.2,1.2],
        vertical_flip=True,
        fill_mode="nearest")

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

model.save('cracks_resnet.h5')

learn_rate = [0.001, 0.01]
batch_size = [10, 20]
epochs = [10, 50]
model = KerasClassifier(build_fn=ModelMobile, verbose=0)
param_grid = dict(learn_rate=learn_rate,
                  batch_size=batch_size, 
                  epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(trainX,trainY)
# train the head of the network
print("[INFO] training head...")

import sklearn.metrics as metrics
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

y_pred_ohe = KerasClassifier.predict(testX)  # shape=(n_samples, 12)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)  # shape=(12, 12)#When the model is trained, 
#Analize plot a graph to see its learning curve.
#Save the model for later use.
#To save the trained model
model.save('new_cracks_model.h5')

model = load_model('new_cracks_model.h5')

# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
yhat_probs.shape
yhat_probs1 =  np.argmax(yhat_probs,axis=1)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)

import numpy as np
rounded_labels=np.argmax(testY, axis=1)
confusion_matrix(rounded_labels,yhat_probs1)
accuracy_score(rounded_labels,yhat_probs1)
precision_score(rounded_labels,yhat_probs1,average='micro')
recall_score(rounded_labels,yhat_probs1,average='micro')
f1_score(rounded_labels,yhat_probs1,average='micro')

#class - 1 - Cracks High
tp = 11
fp = (0+5) = 5
tn = (15+4+1+10) = 30
fn = (3+7) = 10
acc = (11+30)/(11+5+30+10) = 73%

#class - 2 - cracks low

tp = 15
fp = (3+4) = 7
tn = (11+7+5+10) = 33
fn = (0+1) = 1
acc = (15+33)/(15+7+33+1) = 85%
#class 3 - cracks medium

tp = 10
fp = (7+1) = 8
tn = (11+0+3+15) = 29
fn = (5+4) = 9
acc  = (10+29)/(10+8+29+9) = 69%
