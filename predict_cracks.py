# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 08:12:33 2021

@author: Manoj
"""

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression

import sys
#sys.path.append('')

from nms_user import non_max_suppression_slow
from helpers import sliding_window
import glob

model = load_model('/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/models/cracks22.h5')

#path = "./cracks_test/*.*"
path = '/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/cracks_test/*.*'
for num,inputpath in enumerate(glob.glob(path)):
    
image = load_img(path, target_size=(224, 224))

image = img_to_array(image)
image = preprocess_input(image)

img = np.reshape(image, [1,224,224,3])
classes = model.predict(img)
print(classes)

if(classes[0][0] > 0.65):
    print("image is a low crack")
    (winW, winH) = (64,64)
    
    step = 16
    
    cv_image = cv2.imread(path)
    cv_image_resized = cv2.resize(cv_image,(224,224))
    #cv2.imshow("resized",cv_image_resized)
    #cv2.waitKey(0)
    np_array_boxes = np.array([(0,0,0,0)])
    prob = np.array([0])
    image_to_draw = np.array
    image_to_redraw = np.array
    
    for index,(x,y,window) in enumerate(sliding_window(image = cv_image_resized, stepSize = step, windowSize = (winW,winH))):
        #np_array_boxes = np.array([(0,0,0,0)])
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
               
        t_rect = (x,y,winW,winH)
        #cropped_image = cv_image_resized[y:winH,x:winW]
        #print(cropped_image.shape)
        #print(cropped_image)
        #cv2.imshow("cropped",cropped_image)
        #cv2.waitKey(0)
        #break
        #window = cv2.GaussianBlur(window,(11,11),1.5)
        
        win = cv2.resize(window,(224,224))
        #cv2.imshow("reshaped",win)
        #cv2.waitKey(0)
        image_arr = img_to_array(win)
        image_processed = preprocess_input(image_arr)
        image_reshaped = np.reshape(image_arr, [1,224,224,3])
        
        new_classes = model.predict(image_reshaped)
       
        print(new_classes)
       # break
        if(new_classes[0][0]>0.81):
            np_array_boxes = np.append(np_array_boxes,t_rect)
            prob = np.append(prob,new_classes[0][1])
        
    
    image_to_draw = cv_image_resized.copy()
    image_to_redraw = cv_image_resized.copy()
    
    
    if(len(np_array_boxes) > 1):
        boxes = np_array_boxes.reshape(len(np_array_boxes)//4,4)
        for index,rect in enumerate(boxes):
            (x,y,winW,winH) = rect
            cv2.rectangle(image_to_draw,(x,y),(x + winW,y+winH),(0,0,255),2)
        
        cv2.imwrite('/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/Cracks_Output/image_drawmn.jpg',image_to_draw)
    
        #boxes = non_max_suppression_slow(boxes_reshaped[1:-1],0.4)
        boxes = non_max_suppression(boxes,prob,0.3)
        boxes = non_max_suppression(boxes,None,0.2)
    
        for box in boxes:
            (x,y,winW,winH) = box.astype('int')
            cv2.rectangle(image_to_redraw,(x,y),(x + winW,y+winH),(0,255,0),2)
        
        cv2.imwrite('/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/Cracks_Output/image_redrawmn.jpg',image_to_redraw)
        
    else:
        print("no boxes found")

elif(classes[0][1]>0.65 ):
    print("image is a medium crack")
elif(classes[0][2]>0.65 ):
    print("image is a high crack")
    #cv_image = cv2.imread('')
    #cv2.imshow(cv_image)
    #cv2.waitKey(5000)
else:
    print("confilicting image")

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title('processed')

plt.show()

