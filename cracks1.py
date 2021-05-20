from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
import glob

#make sure model's path is at correct location
#cracks_model = load_model('cracks_model.h5')
cracks_model = load_model('/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/models/cracks_resnet.h5')
path = "/home/smohammad/Projects/Raj_Project/Flexible1/Cracks/cracks_test/*.*"
for num,inputpath in enumerate(glob.glob(path)):
    #enter the image path
    img3 = cv2.imread(inputpath,cv2.IMREAD_COLOR)
    imgC_2 = cv2.resize(img3,(224,224))
    img3 = img_to_array(imgC_2)
    img3  = preprocess_input(img3)
    img3 = np.reshape(img3,[1,224,224,3])
    
    classes1 = cracks_model.predict(img3)
    classes1 = classes1.tolist()
    print(classes1)
    flat_list = []
    for sublist in classes1:
        for item in sublist:
            flat_list.append(item)
    
    labels = ['high','low','medium']
    zipped_list = list(list(x) for x in zip(labels,flat_list))
    res = sorted(zipped_list,key = lambda x: x[1])
    text = str(res[-1][0])
    print(text)
    
    font = cv2.FONT_HERSHEY_COMPLEX
    bottomLeft = (30,40)
    fontscale = 1
    fontcolor = (0,0,255)
    linetype = 2
    
    #type(image)
    img_2 = cv2.resize(imgC_2,(224,224))
    cv2.putText(img_2,text,bottomLeft,font,fontscale,fontcolor,linetype)
    if(text == 'low'):
        cv2.rectangle(img_2,(1,1),(200,200),(0,255,0),2,2)
    elif(text == 'medium'):
        cv2.rectangle(img_2,(1,1),(200,200),(255,0,0),2,2)
    elif(text=='high'):
        cv2.rectangle(img_2,(1,1),(200,200),(0,0,255),2,2)
    else:
        pass
    
    img_concate_Hori=np.concatenate((imgC_2,img_2),axis=1)
    #cv2.imshow('concatenated_Hori',img_concate_Hori)
    #cv2.imshow("image with text ",img_ph2)
    #cv2.imwrite("D:/Road_Project_DIP/Cracks/Output/cracks{}.jpg".format(num),img_concate_Hori)
    
    #cv2.imshow("image with text ",img_concate_Hori)
    #cv2.imwrite("./Cracks_Output/cracks_{}{}.jpg".format(text,num),img_concate_Hori)
    cv2.imwrite("./Cracks_Output/cracks{}.jpg".format(num),img_concate_Hori)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

