from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from imutils import paths
import os
from shutil import rmtree
import glob
from sklearn.metrics import confusion_matrix,classification_report

potholes_model = load_model('potholes_model.h5')

#InputImages = list(paths.list_images('./Images/'))
path = "./Input/*.*"
classes2 = []
for num,InputImage in enumerate (glob.glob(path)):

    img_ph = cv2.imread(InputImage,cv2.IMREAD_COLOR)
    img_ph2 = cv2.resize(img_ph,(224,224))
    img_ph_arr = img_to_array(img_ph2)
    img_ph_prep  = preprocess_input(img_ph_arr)
    img_ph_res = np.reshape(img_ph_prep,[1,224,224,3])
    
    classes1 = potholes_model.predict(img_ph_res)
    classes2.append(classes1)
predicted_classes = np.argmax(classes2, axis=1)
#print(classes2)
    #classes1 = classes1.tolist()
    
        #print(classes1)
    flat_list = []
    for sublist in classes1:
        for item in sublist:
            flat_list.append(item)

    labels = ['Not a Pothole','Pothole']
    zipped_list = list(list(x) for x in zip(labels,flat_list))
    res = sorted(zipped_list,key = lambda x: x[1])
    text = str(res[-1][0])
        #print(text)

    font = cv2.FONT_HERSHEY_COMPLEX
    bottomLeft = (40,60)
    fontscale = 1
    fontcolor = (75,75,75)
    linetype = 2

        #type(image)
    img_2 = cv2.resize(img_ph2,(224,224))
    cv2.putText(img_2,text,bottomLeft,font,fontscale,fontcolor,linetype)
    cv2.rectangle(img_2,(1,1),(224,224),(0,255,0),2,2)
    #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        #img_ph2 = cv2.imshow("image with text ",img_ph2)
    #cv2.imshow("image with text ",img_ph2)
        #img_v = cv2.vconcat(img_ph,img_ph2)
        #cv2.imshow("image with input & output ",img_v)
    img_concate_Hori=np.concatenate((img_ph2,img_2),axis=1)
        #concatanate image Vertically
        #img_concate_Verti=np.concatenate((img_ph,img_ph2),axis=0)
    #cv2.imshow('concatenated_Hori',img_concate_Hori)
        #cv2.imshow('concatenated_Verti',img_concate_Verti)
    #cv2.imwrite("./Output/potholes{}.jpg".format(num),img_concate_Hori)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

classes1

#number of inputs
#number of potholes detected TP
test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
     './Input/',target_size=(224, 224),batch_size=4,shuffle=False)
test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = potholes_model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)#
true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
report = confusion_matrix(true_classes, predicted_classes, target_names=class_labels)
print(report)   
