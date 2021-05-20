import cv2
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

infile = pd.read_csv("train.csv")
print(infile.head(6))

data = []
labels = []

for record in range(0, len(infile)):
    path = infile["images"][record]
    label= infile["labels"][record]
    #reading images from the path
    print("Reading file: " + path)
    image = load_img(path,target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    labels.append(label)
    
data = np.array(data, dtype="float32")
labels = np.array(labels)
print('Labels found: ' + str(labels))
np.unique(labels)

testfile = pd.read_csv("test.csv")

#The next step is to load the pre-trained model and customize it according to our problem.
#This technique is named Transfer learning
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_shape=(224, 224, 3))

print('baseModel: ' + str(baseModel))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(7, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

#Convert the labels into one-hot encoding. 
#Split the data into training and testing sets to evaluate them.
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
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
EPOCHS = 20      
BS = 32
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#collecting the test data
test = []
for record in range(0, len(infile)):
    path = testfile["images"][record]
    #reading images from the path
    print("Reading file: " + path)
    image = load_img(path,target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    test.append(image)
    
test = np.array(data, dtype="float32")

#predict the model with new data
y_pred = model.predict(test)

#measuring the accuracy of the model
score = 100*f1_score(testY,y_pred,average='weighted')


