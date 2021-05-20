from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np
import cv2
import glob
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils.object_detection import non_max_suppression
from nms_user import non_max_suppression_slow
from helpers import sliding_window
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

potholes_model = load_model('/home/smohammad/Projects/Raj_Project/Pothole_New/model/pothole_model_224.h5')

#InputImages = list(paths.list_images('./Images/'))
path = "/home/smohammad/Projects/Raj_Project/Pothole_New/inputs/IMG_20201115_070153.jpg"

#for num,InputImage in enumerate (glob.glob(path)):

image = load_img(path,target_size=(224,224))
image = img_to_array(image)
image = preprocess_input(image)
image= np.expand_dims(image,axis=0)
result = potholes_model.predict(image)

if result[0][0]>0.65:
    print('Pothole')    

    
else:
    print('Normal')


plt.figure(figsize=(10,10))
plt.imshow(image)
plt.title('processed')

plt.show()


#classes2.append(classes1)
flat_list = []
for sublist in result:
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
img_2 = cv2.resize(test_image,(64,64))
cv2.putText(img_2,text,bottomLeft,font,fontscale,fontcolor,linetype)
cv2.rectangle(img_2,(1,1),(64,64),(0,255,0),2,2)
img_concate_Hori=np.concatenate((test_image,img_2),axis=1)
#cv2.imshow('concatenated_Hori',img_concate_Hori)
cv2.imwrite("/home/smohammad/Projects/Raj_Project/Pothole_New/outputs/potholes{}.jpg".format(num),img_concate_Hori)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
