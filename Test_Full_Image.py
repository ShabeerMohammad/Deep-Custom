import os
import cv2
import numpy as np
from autoencodermodel import autoencoder_model
from shutil import rmtree

Test_Images_Path = "Ravelling"
Ravelling_Out = Test_Images_Path+"_out"
if os.path.isdir(Ravelling_Out):
    rmtree(Ravelling_Out)
    os.mkdir(Ravelling_Out)
else:
    os.mkdir(Ravelling_Out)
    
Im_Size = 64
Slide = 64
thr = 20
ImagesList = os.listdir(Test_Images_Path)
autoencoder = autoencoder_model(Im_Size)
for Image in ImagesList:
    print("Applying:"+Image)
    ImName = os.path.join(Test_Images_Path, Image)
    test_image = cv2.imread(ImName,cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (Im_Size*6, Im_Size*6))
    test_image_255 = test_image/255
    Height, Width = test_image.shape

    count = 0
    for i in range(0, Height, Slide):
        for j in range(0, Width, Slide):
            if (i>=0) and (j>=0) and ((i+Im_Size)<=Height) and ((j+Im_Size)<=Width):
                count = count+1

    patches = np.zeros((count, Im_Size, Im_Size,1))
    count = 0
    for i in range(0, Height, Slide):
        for j in range(0, Width, Slide):
            if (i>=0) and (j>=0) and ((i+Im_Size)<=Height) and ((j+Im_Size)<=Width):
                patches[count, :, :, 0] = test_image_255[i:i+Im_Size, j:j+Im_Size]
                count = count+1

    decoded_patches = (autoencoder.predict(patches)*255).astype('uint8')

    decoded_image = np.zeros((Height, Width), dtype = 'uint8')
    count = 0
    for i in range(0, Height, Slide):
        for j in range(0, Width, Slide):
            if (i>=0) and (j>=0) and ((i+Im_Size)<=Height) and ((j+Im_Size)<=Width):
                decoded_image[i:i+Im_Size, j:j+Im_Size] = decoded_patches[count, :, :, 0]
                count = count+1
#                cv2.imshow(str(count), decoded_image)


    diff_image = np.abs(decoded_image.astype('float32') - test_image.astype('float32'))
 #   print(diff_image)
    diff_image_255 = diff_image
 #   print(diff_image_255)
    mask = ((diff_image>thr)*255).astype('uint8')

    final_image = np.zeros((Height, Width*4,3), dtype = 'uint8')
    final_image[0:Height, 0:Width,0] = test_image
    final_image[0:Height, 0:Width,1] = test_image
    final_image[0:Height, 0:Width,2] = test_image
    final_image[0:Height, Width:2*Width,0] = decoded_image
    final_image[0:Height, Width:2*Width,1] = decoded_image
    final_image[0:Height, Width:2*Width,2] = decoded_image
    final_image[0:Height, 2*Width:3*Width,0] = diff_image_255
    final_image[0:Height, 2*Width:3*Width,1] = diff_image_255
    final_image[0:Height, 2*Width:3*Width,2] = diff_image_255
    final_image[0:Height, 3*Width:4*Width,0] = mask
    final_image[0:Height, 3*Width:4*Width,1] = mask
    final_image[0:Height, 3*Width:4*Width,2] = mask

    ravelling_strength = np.sum((mask.astype('float32')/255))/(Height*Width)
    ravelling_area = np.sum((mask.astype('float32')/255))
    final_image = cv2.putText(final_image, 'Ravelling_Strength:'+str(np.round(ravelling_strength,3)), (20,20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)

    final_image = cv2.putText(final_image, 'Ravelling_Area:'+str(ravelling_area), (20,70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)
    
    final_image = cv2.resize(final_image, (Im_Size*3*4, Im_Size*3))
    
    cv2.imwrite(os.path.join(Ravelling_Out, Image), final_image)
#    cv2.imshow('a', final_image)
#    cv2.waitKey(0)
                
                

