from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,load_img
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

cracks_model = load_model('/home/smohammad/Projects/Raj_Project/Cracks_New/model/cracks_model24.h5')

#InputImages = list(paths.list_images('./Images/'))
path = "/home/smohammad/Projects/Raj_Project/Cracks_New/Images/normal/1.jpg"

#for num,InputImage in enumerate (glob.glob(path)):

image1 = cv2.imread(path,cv2.IMREAD_COLOR)
image2 = cv2.resize(image1,(224,224))
image3 = img_to_array(image2)
image4 = preprocess_input(image3)
image5 = np.reshape(image4,[1,224,224,3])
result = cracks_model.predict(image5)

test_image = load_img(path,target_size=(224,224))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cracks_model.predict(test_image)
if result[0][0]==1:
    text = 'Normal'
else:
    text = 'Crack'
print(text)

font = cv2.FONT_HERSHEY_COMPLEX
bottomLeft = (40,60)
fontscale = 1
fontcolor = (0,0,205)
linetype = 2
#type(image)
#img_2 = cv2.resize(image2,(224,224))
#image1 = np.float32(image1)
img_2 = cv2.resize(image2,(224,224))
cv2.putText(img_2,text,bottomLeft,font,fontscale,fontcolor,linetype)
if (text=='Normal'):
    cv2.rectangle(image1,(1,1),(200,200),(0,0,205),2,2)
elif (text=='Crack'):
    cv2.rectangle(image1,(1,1),(200,200),(255.0,0),2,2)
#else:
#    pass

img_concate_Hori=np.concatenate((image2,img_2),axis=1)
cv2.imshow('concatenated_Hori',img_concate_Hori)
#cv2.imwrite("/home/smohammad/Projects/Raj_Project/Cracks_New/outputs/normal1.jpg",img_concate_Hori)
cv2.waitKey(0)
cv2.destroyAllWindows()


