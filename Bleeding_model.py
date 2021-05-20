
                    
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

Bleed_imagePaths = list(paths.list_images('./Bleeding/'))
print('Loading ' + str(len(Bleed_imagePaths )) + ' images...')

data_bleed = []
labels_bleed = []

# loop over the image paths
for imagePath in Bleed_imagePaths:    
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    #load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    
    image = img_to_array(image)
    image = preprocess_input(image)
    # update the data and labels lists, respectively
    data_bleed.append(image)
    labels_bleed.append(label)
# convert the data and labels to NumPy arrays
data_bleed = np.array(data_bleed, dtype="float32")
labels_bleed = np.array(labels_bleed)

#The next step is to load the pre-trained model and customize it according to our problem.
#This technique is named Transfer learning
baseModel_bleed = MobileNetV2(weights="imagenet", include_top=False,input_shape=(224, 224, 3))

# construct the head of the model that will be placed on top of the
# the base model
headModel_bleed = baseModel_bleed.output
headModel_bleed = AveragePooling2D(pool_size=(7, 7))(headModel_bleed)
headModel_bleed = Flatten(name="flatten")(headModel_bleed)
headModel_bleed = Dense(128, activation="relu")(headModel_bleed)
headModel_bleed = Dense(128, activation="relu")(headModel_bleed)
#headModel = Dropout(0.5)(headModel)
headModel_bleed = Dropout(0.3)(headModel_bleed)
headModel_bleed = Dense(1, activation="sigmoid")(headModel_bleed)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel_bleed.input, outputs=headModel_bleed)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel_bleed.layers:
	layer.trainable = False

#Convert the labels into one-hot encoding. 
#Split the data into training and testing sets to evaluate them.
lb = LabelEncoder()
labels_bleed = lb.fit_transform(labels_bleed)
#labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data_bleed, labels_bleed,
	test_size=0.20, stratify=labels_bleed, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#Compile the model and train it on the augmented data.
INIT_LR = 1e-4   #learning rate
EPOCHS =100       
BS = 8
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#When the model is trained, 
#Analize plot a graph to see its learning curve.
#Save the model for later use.
#To save the trained model
model.save('bleeding_model.h5')

#predictions
image = load_img('./Cracks/Low/IMG_20201115_063617.jpg', target_size=(224, 224))

image = img_to_array(image)
image = preprocess_input(image)

plt.figure(figsize = (3,3))
plt.imshow(image)
plt.title("testImage")

img = np.reshape(image,[1,224,224,3])
classes = model.predict(img)
print(classes)


image1 = load_img('./Pot holes/IMG_20201115_063202.jpg', target_size=(224, 224))
image1 = img_to_array(image1)
image1 = preprocess_input(image1)

image1 = np.reshape(image1,[1,224,224,3])
classes1 = model.predict(image1)
print(classes1)


image1 = load_img('1.jpeg', target_size=(224, 224))
image1 = img_to_array(image1)
image1 = preprocess_input(image1)

image1 = np.reshape(image1,[1,224,224,3])
classes1 = model.predict(image1)
print(classes1)


image1 = load_img('2.jpeg', target_size=(224, 224))
image1 = img_to_array(image1)
image1 = preprocess_input(image1)

image1 = np.reshape(image1,[1,224,224,3])
classes1 = model.predict(image1)
print(classes1)

# evaluate the model
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
_, test_acc = model.evaluate(testX, testY, verbose=0)


# predict probabilities for test set
yhat_probs = model.predict(testX, verbose=0)
yhat_probs.shape
yhat_probs1 =  np.argmax(yhat_probs,axis=1)
# predict crisp classes for test set
yhat_classes = model.predict_classes(testX, verbose=0)

# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(testY, yhat_probs1)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(testY, yhat_probs1,average='micro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(testY, yhat_probs1,average='micro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(testY, yhat_probs1,average='micro')
print('F1 score: %f' % f1)

confusion_matrix(testY,yhat_probs1)

#class - 1
tp = 8
tn = (11+6+10+4) = 31
fp = 10+2 = 12
fn = 6+7 = 13

p = 8/(8+12)
r = 8/(8+13)

#class - 2
tp = 11
tn = ()