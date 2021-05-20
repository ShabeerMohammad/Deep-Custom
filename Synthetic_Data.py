
import numpy as np
import cv2

ImSize = 64
count = 0
print('a')
for trail in range(1,100):
    for Intensity in range(100,250,5):
        Imagematrix = Intensity*np.ones((ImSize, ImSize), dtype = 'uint8')
        for Var in range(5,15):
            noisematrix = np.random.normal(0,Var,ImSize*ImSize)
            noisematrix = np.reshape(noisematrix,[ImSize, ImSize]).astype('uint8')
            Imagematrix = Imagematrix+noisematrix

            cv2.imwrite("SynImages/"+str(count)+'.jpg', Imagematrix)
            count = count+1
