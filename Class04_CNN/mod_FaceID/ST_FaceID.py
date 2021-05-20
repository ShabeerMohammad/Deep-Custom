'''
Created on 26-Aug-2020
@author: José Arturo Gaona
'''
#########################################################################################
#  COPYRIGHT 2020
#  \verbatim
#                 This software is copyright protected and proprietary to Condumex.
#                 All other rights remain with Condumex.
#  \endverbatim
#  LICENSE
#          Module: **EDT_AccessCTRL/Gateway
#          Description: **This module provide all methods to train, add, remove or identify faceIDs from a givem model file path
#          Enterprise: Condumex
#          SW Developer: **José Arturo Gaona Cuadra
#          
#          File: **ST_FaceID.py
#          Feature: **EDT_AccessCTRL
#          Design:  **Diagrama_Clases_ST_FaceID_v1.0.ppt
#          Deviations: **Por aclarar con Calidad
#   
#  **Information that must change according to the script
#########################################################################################
_debugTest=False
_TestWithTensorFlow = False   #DBL_40
import os
try:
    import mod_FaceID.ST_FaceID_cfg as fid_cfg
    parentFolder =  '/mod_FaceID/'
    pass
except Exception as e:
    import ST_FaceID_cfg as fid_cfg
    parentFolder =  '/'
    pass

import unittest
import face_recognition
import cv2
import os
import pickle
import numpy as np
import urllib.request                       # DBL_41
from urllib.request import Request, urlopen # DBL_41
from PIL import Image

class ST_FaceID:
    '''
    Static interface for Image recognition
    '''
    __instance = None
    UseTensorFlow = True

    @staticmethod
    def getInstance():
        """ 
            Static access method.
        """
        if ST_FaceID.__instance is None:
            ST_FaceID()
        return ST_FaceID.__instance

    def __init__(self,useTensorFlow=True):  #DBL_40
        """ 
            Virtually private constructor. 
        """
        ST_FaceID.UseTensorFlow=useTensorFlow
        if ST_FaceID.__instance != None:
            raise Exception(fid_cfg.SINGLETONE_MODEL_EXCEPTION)
        else:
            ST_FaceID.__instance = self
            ST_FaceID.Model = {fid_cfg.MODEL_FACE_LIST:[], fid_cfg.MODEL_FACE_ENCODE_LIST:[]}
            ST_FaceID.FirstIdentifyFace()   # DBL_24
            if (ST_FaceID.UseTensorFlow==True):    #DBL_40
                ST_FaceID.tensorflow = __import__('tensorflow')
                modelPath = fid_cfg.MASK_MODEL
                if (os.path.exists(modelPath)):
                    ST_FaceID.MaskModel = ST_FaceID.tensorflow.keras.models.load_model(modelPath)
                    ST_FaceID.FirstMaskDetection()   # DBL_24
                else:
                    raise Exception(fid_cfg.NO_MODEL_EXCEPTION + fid_cfg.MASK_MODEL)
    @staticmethod   # DBL_24
    def FirstIdentifyFace():
        frame = face_recognition.load_image_file(os.path.dirname(os.path.abspath(__file__)) + '/UnitTestImages/TestMeganFox.jpg')
        face_encoding = face_recognition.face_encodings(frame)[0]
        result = ST_FaceID.IdentifyFace(face_encoding)

    @staticmethod # DBL_24
    def FirstMaskDetection():
        frame = face_recognition.load_image_file(os.path.dirname(os.path.abspath(__file__)) + '/UnitTestImages/TestMeganFox.jpg')
        result = ST_FaceID.GetMaskStatus(frame)
           
    @staticmethod
    def Load(modelFilePath):
        '''
            Loads the model from a file path
            usage example: 

            import ST_FaceID
            faceIdentification = ST_FaceID.getInstance()
            result = faceIdentification.Load('unitestModel.pkl')
        '''
        result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Model was loaded on ST_FaceID.Model'}
        try:
            with open(modelFilePath, 'rb') as f:
                ST_FaceID.Model = pickle.load(f)
        except Exception as e:
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:fid_cfg.GENERAL_RESULT_EXCEPTION + str(e)}
        return result

    @staticmethod
    def SaveModel(modelFilePath):
        '''
            Save ST_FaceID.Model to a file
        '''
        result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Model was saved on ' + modelFilePath}
        try:
            with open(modelFilePath, 'wb') as f:
                pickle.dump(ST_FaceID.Model, f)
        except Exception as e:
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT: fid_cfg.GENERAL_RESULT_EXCEPTION + str(e)}
        return result

    @staticmethod
    def AddFaceIDs(modelFilePath,objectsToTrainList):
        '''
            How to use:
            faceIdentification = ST_FaceID.getInstance()
            objectsToTrainList = []
            objectsToTrainList.append({'faceID':'MeganFox','path':'./TestCaseImages/MeganFox.jpg'})
            result = faceIdentification.AddFaceIDs('modelFilePath.pkl',objectsToTrainList)
            
            result--> [{fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'MeganFox was added to ST_FaceID.Model'}]
        '''
        resultList = []
        for objectsToTrain in objectsToTrainList:
            try:
                #result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT: objectsToTrain['faceID'] + ' was added to ST_FaceID.Model'}
                result = {fid_cfg.GENERAL_RESULT_USER_ID: objectsToTrain['faceID'],
                            fid_cfg.GENERAL_RESULT_STATUS:True,
                            fid_cfg.GENERAL_RESULT_COMMNENT: objectsToTrain['faceID'] + ' was added to ST_FaceID.Model'}
                if (ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST])==[] or (objectsToTrain['faceID'] not in ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST]):
                    # DBL_41
                    if (('https:' in str(objectsToTrain['path'])) or ('http:' in str(objectsToTrain['path']))):
                        url = str(objectsToTrain['path'])
                        req = Request(url, headers={'User-Agent': 'XYZ/3.0'})
                        image = Image.open(urllib.request.urlopen(req, timeout=10))
                        image = np.asarray(image)
                    else:
                        # DBL_41
                        image = face_recognition.load_image_file(objectsToTrain['path'])
                    #face_locations = face_recognition.face_locations(image, model='cnn')  # DBL_21  improve accuracy using CNN model instead of HUG
                    face_locations = face_recognition.face_locations(image)  # DBL_21  improve accuracy using CNN model instead of HUG
                    face_encode = face_recognition.face_encodings(image)[0]
                    ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST].append(objectsToTrain['faceID'])
                    ST_FaceID.Model[fid_cfg.MODEL_FACE_ENCODE_LIST].append(face_encode)
                else:
                    #result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT: objectsToTrain['faceID'] + ' was Already in the ST_FaceID.Model'}
                    result = {fid_cfg.GENERAL_RESULT_USER_ID: objectsToTrain['faceID'],
                            fid_cfg.GENERAL_RESULT_STATUS:False,
                            fid_cfg.GENERAL_RESULT_COMMNENT: objectsToTrain['faceID'] + ' was already in ST_FaceID.Model'}
            except Exception as e:
                #result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT: fid_cfg.GENERAL_RESULT_EXCEPTION + str(e)}
                result = {fid_cfg.GENERAL_RESULT_USER_ID: objectsToTrain['faceID'],
                            fid_cfg.GENERAL_RESULT_STATUS:False,
                            fid_cfg.GENERAL_RESULT_COMMNENT: fid_cfg.GENERAL_RESULT_EXCEPTION + str(e)}
            resultList.append(result)
        if len(resultList)>0:
            ST_FaceID.SaveModel(modelFilePath)
        return resultList

    @staticmethod
    def Train(pathImages,modelFilePath,extension=fid_cfg.FACE_IMAGE_FORMAT):
        '''
            Train the Face Racognition model 
                pathImages = Path where the faces are placed
                modelFilePath = Model file path where the new updated model will be saved
                Extension = types  list of file to be used on filter []
                Returns the result status list 
            Usage example: 

            import ST_FaceID

            faceIdentification = ST_FaceID.getInstance()
            result = faceIdentification.Train('./TestCaseImages','unitestModel.pkl',['jpg'])
        '''
        result = [{fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Model was trained from:'+ pathImages +' and saved on ' + modelFilePath}]
        if not os.path.exists(modelFilePath):
            ST_FaceID.SaveModel(modelFilePath)
        loadResult = ST_FaceID.Load(modelFilePath)

        if loadResult[fid_cfg.GENERAL_RESULT_STATUS]:
            faceIDList = []
            faceEncodeList = []
            objectsToTrainList = []
            for root, dirs, files in os.walk(pathImages):
                for file in files:
                    if (file.endswith(extension[0])):
                        path = os.path.join(root, file)
                        faceID = os.path.basename(path).replace('.' + extension[0], '')
                        if (faceID not in ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST]):
                            objectsToTrainList.append({'faceID':faceID,'path':path})
            if (len(objectsToTrainList)>0):
                result = ST_FaceID.AddFaceIDs(modelFilePath,objectsToTrainList)
        else:
            result = loadResult
        return result

    @staticmethod
    def GetMaskStatus(crop_image):
        '''
            Determina if the face image parameter is using a face-mask or not
            How to use it:
            faceIdentification = ST_FaceID.getInstance()
            frame = face_recognition.load_image_file('./UnitTestImages/Mask.jpg')
            result = face_recognition.GetMaskStatus(frame)
            result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Mask status was determied as ' + label,'maskStatus':False,'accuracy':100.0}
        '''
        if (ST_FaceID.UseTensorFlow==True):   #DBL_40
            label = "No_Mask"
            result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Mask status was determied as ' + label,'maskStatus':False,'accuracy':100.0}
            face_frame = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            face_frame = cv2.resize(face_frame, (224, 224))
            face_frame = ST_FaceID.tensorflow.keras.preprocessing.image.img_to_array(face_frame)
            face_frame = np.expand_dims(face_frame, axis=0)
            face_frame =  ST_FaceID.tensorflow.keras.applications.mobilenet_v2.preprocess_input(face_frame)
            faces_list=[]
            faces_list.append(face_frame)
            if len(faces_list)>0:
                preds = ST_FaceID.MaskModel.predict(faces_list)
            #mask contain probabily of wearing a mask and vice versa
            for pred in preds:
                (mask, withoutMask) = pred

            accuracy = max(mask, withoutMask) * 100

            if mask > withoutMask:
                label = "Mask"
                result['maskStatus'] = True
                result[fid_cfg.GENERAL_RESULT_COMMNENT] = 'Mask status was determied as ' + label
            result['accuracy'] = accuracy
        else:  #DBL_40
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:'Use tensorflow is not supported for this time','maskStatus':False,'accuracy':0.0}
        return result

    @staticmethod
    def IdentifyFace(face_encoding,unknown=fid_cfg.UNKNOWN_USERID,tolerance=fid_cfg.FACE_RECOG_TOLERANCE_NO_MASK):
        '''
            Identify only one face in the frame face encoded
            how to use it:

            faceIdentification = ST_FaceID.getInstance()
            faceIdentification.Load('unitestModel.pkl')
            frame = face_recognition.load_image_file('./UnitTestImages/TestMeganFox.jpg')
            face_encoding = face_recognition.face_encodings(frame)[0]
            result = faceIdentification.IdentifyFace(face_encoding)
            #result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'face image was found on ST_FaceID.Model','faceID':faceID}

        '''
        try:
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:'face image was NOT found on ST_FaceID.Model','faceID':unknown}
            matches = face_recognition.compare_faces(ST_FaceID.Model[fid_cfg.MODEL_FACE_ENCODE_LIST],face_encoding,tolerance)
            if True in matches:
                first_match_index = matches.index(True)
                faceID = ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST][first_match_index]
                result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'face image was found on ST_FaceID.Model','faceID':faceID}
        except Exception as e:
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:fid_cfg.GENERAL_RESULT_EXCEPTION+ str(e),'faceID':unknown}        
        return result

    @staticmethod
    def RemoveFaceID(modelFilePath,faceID):
        '''
            How to use:
            faceIdentification = ST_FaceID.getInstance()
            result = faceIdentification.RemoveFaceID('modelFilePath.pkl','MeganFox')
            result--> {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'MeganFox was removed from ST_FaceID.Model'}]
        '''
        try:
            result = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT: faceID + ' was removed from the ST_FaceID.Model'}
            if faceID in ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST]:
                faceID_index = ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST].index(faceID)
                ST_FaceID.Model[fid_cfg.MODEL_FACE_LIST].remove(faceID)
                del(ST_FaceID.Model[fid_cfg.MODEL_FACE_ENCODE_LIST][faceID_index])
                ST_FaceID.SaveModel(modelFilePath)
            else:
                result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT: faceID + ' was not found on the ST_FaceID.Model'}
        except Exception as e:
            result = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:fid_cfg.GENERAL_RESULT_EXCEPTION+ str(e),'faceID':unknown}        
        return result

    @staticmethod
    def UpdateFaceIDs(modelFilePath,objectsToTrainList):
        '''
            How to use:
            faceIdentification = ST_FaceID.getInstance()
            objectsToTrainList = []
            objectsToTrainList.append({'faceID':'MeganFox','path':'./TestCaseImages/MeganFox.jpg'})
            result = faceIdentification.UpdateFaceIDs('modelFilePath.pkl',objectsToTrainList)
            result--> [{fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'MeganFox was added on ST_FaceID.Model'}]
        '''
        resultList = []
        try:
            #Delete all FaceIDs from the model
            for objectsToTrain in objectsToTrainList:
                ST_FaceID.RemoveFaceID(modelFilePath,objectsToTrain['faceID'])
            resultList = ST_FaceID.AddFaceIDs(modelFilePath,objectsToTrainList)
        except Exception as e:
            resultList.append({fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:fid_cfg.GENERAL_RESULT_EXCEPTION+ str(e),'faceID':unknown})
        return resultList

'''
**********************************
********* Unit Test **************
*** https://docs.python.org/2/library/unittest.html#assert-methods *****
**********************************
'''
class TC001_Test_SingleTone(unittest.TestCase):
   @unittest.skipIf(_TestWithTensorFlow==True,"DebugMode")
   def test001_1A_TestSingletoneException(self):
      print('.******************** test001_1_SingletoneException useTensorFlow = False ***')
      faceIdentification = ST_FaceID(useTensorFlow=False)  #DBL_40
      exceptionFlag = False 
      try:
         faceIdentificatio2 = ST_FaceID(useTensorFlow=False)   #DBL_40
      except Exception as e:
         exceptionFlag = True
      else:
         pass
      self.assertTrue(exceptionFlag,True)

   @unittest.skipIf(_TestWithTensorFlow==False,"DebugMode")
   def test001_1B_TestSingletoneException(self):
      print('.******************* test001_1_SingletoneException useTensorFlow = True *****')
      faceIdentification = ST_FaceID(useTensorFlow=True)  #DBL_40
      exceptionFlag = False 
      try:
         faceIdentificatio2 = ST_FaceID(useTensorFlow=True)   #DBL_40
      except Exception as e:
         exceptionFlag = True
      else:
         pass
      self.assertTrue(exceptionFlag,True)

   @unittest.skipIf(_debugTest==True,"DebugMode")
   def test001_2_TestMultipleGetInstances(self):
      print('******************* test001_2_MultipleGetInstances useTensorFlow = '+ str(_TestWithTensorFlow) + ' **')
      faceIdentification = ST_FaceID.getInstance() #ST_FaceID(useTensorFlow=_TestWithTensorFlow)
      faceIdentification2 = ST_FaceID.getInstance()
      self.assertEqual(faceIdentification, faceIdentification2)


class TC002_Test_InitializeDB(unittest.TestCase):
    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_01_AddFaceIDs_MeganFox(self):
        print('******************** test002_1_AddFaceIDs_MeganFox *************************')
        faceIdentification = ST_FaceID.getInstance()
        objectsToTrainList = []
        objectsToTrainList.append({'faceID':'MeganFox','path':'./TestCaseImages/MeganFox.jpg'})
        objectsToTrainList.append({'faceID':'LaRoca','path':'./TestCaseImages/LaRoca.jpg'})
        result = faceIdentification.AddFaceIDs('unitestModel.pkl',objectsToTrainList)
        expected=[]
        expected.append({fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'MeganFox was added to ST_FaceID.Model','sap_number': 'MeganFox'})
        expected.append({fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'LaRoca was added to ST_FaceID.Model','sap_number': 'LaRoca'})
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_02_AddFaceIDs_MeganFox(self):
        print('******************** test002_2_AddFaceIDs_MeganFox already added ***********')
        faceIdentification = ST_FaceID.getInstance()
        objectsToTrainList = []
        objectsToTrainList.append({'faceID':'MeganFox','path':'./TestCaseImages/MeganFox.jpg'})
        result = faceIdentification.AddFaceIDs('unitestModel.pkl',objectsToTrainList)
        expected = [{'sap_number': 'MeganFox', 'status': False, 'comment': 'MeganFox was already in ST_FaceID.Model'}]
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_03_Load_Model(self):
        print('******************** test002_3_Load_Model **********************************')
        faceIdentification = ST_FaceID.getInstance()
        result = faceIdentification.Load('unitestModel.pkl')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Model was loaded on ST_FaceID.Model'}
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_04_Load_Model_File_Not_Found(self):
        print('******************** test002_4_Load_Model_File_Not_Found *******************')
        faceIdentification = ST_FaceID.getInstance()
        result = faceIdentification.Load('No_File_Path.pkl')
        self.assertEqual(result['status'],False)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_05_Train_Model(self):
        print('******************** test002_5_Train_Model *********************************')
        faceIdentification = ST_FaceID.getInstance()
        result = faceIdentification.Train('./TestCaseImages','unitestModel.pkl',['jpg'])
        self.assertGreater(len(result),3)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_06_Identify_Face(self):
        print('******************** test002_6_Identify_Face *******************************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'face image was found on ST_FaceID.Model','faceID':'MeganFox'}
        frame = face_recognition.load_image_file('./UnitTestImages/TestMeganFox.jpg')
        face_encoding = face_recognition.face_encodings(frame)[0]
        result = faceIdentification.IdentifyFace(face_encoding)
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_07_RemoveFaceID(self):
        print('******************** test002_7_RemoveFaceID ********************************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')        
        result = faceIdentification.RemoveFaceID('unitestModel.pkl','MeganFox')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT: 'MeganFox' + ' was removed from the ST_FaceID.Model'}
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_08_CheckFaceRemoved(self):
        print('******************** test002_08_CheckFaceRemoved ***************************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT:'face image was NOT found on ST_FaceID.Model','faceID':'0000_0000'}
        frame = face_recognition.load_image_file('./UnitTestImages/TestMeganFox.jpg')
        face_encoding = face_recognition.face_encodings(frame)[0]
        result = faceIdentification.IdentifyFace(face_encoding)
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_09_UpdateFaceIDs_MeganFox(self):
        print('******************** test002_9_UpdateFaceIDs_MeganFox Updated **************')
        faceIdentification = ST_FaceID.getInstance()
        objectsToTrainList = []
        objectsToTrainList.append({'faceID':'MeganFox','path':'./TestCaseImages/MeganFox.jpg'})
        result = faceIdentification.UpdateFaceIDs('unitestModel.pkl',objectsToTrainList)
        expected = [{fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'MeganFox was added to ST_FaceID.Model','sap_number': 'MeganFox'}]
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_10_Savemodel(self):
        print('******************** test002_10_Savemodel **********************************')
        faceIdentification = ST_FaceID.getInstance()
        result = faceIdentification.SaveModel('unitestModel.pkl')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:True,fid_cfg.GENERAL_RESULT_COMMNENT:'Model was saved on ' + 'unitestModel.pkl'}
        del faceIdentification
        self.assertEqual(result,expected)

    @unittest.skipIf(_TestWithTensorFlow==False,"useTensorFlow")  #DBL_40
    def test002_11_GetMaskStatus(self):
        print('******************** test002_11_GetMaskStatus ******************************')
        faceIdentification = ST_FaceID.getInstance()
        frame = face_recognition.load_image_file('./TestCaseImages/Mask.png')
        result = faceIdentification.GetMaskStatus(frame)
        self.assertEqual(result['maskStatus'],True)

    @unittest.skipIf(_TestWithTensorFlow==False,"useTensorFlow")   #DBL_40
    def test002_12_GetMaskStatus(self):
        print('******************** test002_12_GetMaskStatus ******************************')
        faceIdentification = ST_FaceID.getInstance()
        frame = face_recognition.load_image_file('./TestCaseImages/No_Mask.jpg')
        result = faceIdentification.GetMaskStatus(frame)
        self.assertEqual(result['maskStatus'],False)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_13_RemoveFaceID(self):
        print('******************** test002_13_RemoveFaceID *******************************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')        
        result = faceIdentification.RemoveFaceID('unitestModel.pkl','UserID_DoesNot_Exist')
        expected = {fid_cfg.GENERAL_RESULT_STATUS:False,fid_cfg.GENERAL_RESULT_COMMNENT: 'UserID_DoesNot_Exist was not found on the ST_FaceID.Model'}
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_14_FaceIDWithMask(self):
        print('******************** test002_14_IdentifyWithMask ***************************')
        faceIdentification = ST_FaceID.getInstance()
        objectsToTrainList = []
        objectsToTrainList.append({'faceID':'ArturoGaona','path':'./MaskIDTest/2161_Reference.jpg'})
        result = faceIdentification.AddFaceIDs('unitestModel.pkl',objectsToTrainList)
        frame = face_recognition.load_image_file('./MaskIDTest/2161_Mask.jpg')
        face_encoding = face_recognition.face_encodings(frame)[0]
        result = faceIdentification.IdentifyFace(face_encoding)
        expected = {'status': True, 'comment': 'face image was found on ST_FaceID.Model', 'faceID': 'ArturoGaona'}
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_15_AddFaceIDsFromUrl(self):         # DBL_41
        print('******************** test002_15_AddFaceIDsFromUrl **************************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')        
        objectsToTrainList = []
        objectsToTrainList.append({'faceID':'personFromUrl','path':'https://thispersondoesnotexist.com/image'})
        result = faceIdentification.AddFaceIDs('unitestModel.pkl',objectsToTrainList)
        expected = [{'status': True, 'comment': 'personFromUrl was added to ST_FaceID.Model','sap_number': 'personFromUrl'}]
        self.assertEqual(result,expected)

    @unittest.skipIf(_debugTest==True,"DebugMode")
    def test002_16_AddFacesFromUrlAndLocal(self):       # DBL_41
        print('******************** test002_16_AddFacesFromUrlAndLocal ********************')
        faceIdentification = ST_FaceID.getInstance()
        faceIdentification.Load('unitestModel.pkl')        
        faceIdentification.RemoveFaceID('unitestModel.pkl','ArturoGaona')
        objectsToTrainList = []
        # To execute This Unittest
        # Ensure that you have the infraestructure to be able to read the http://192.168.1.93/urlshare/2161_No_Mask.jpg
        objectsToTrainList.append({'faceID':'UrlArturoGaona','path':'http://192.168.1.93/urlshare/2161_No_Mask.jpg'})
        objectsToTrainList.append({'faceID':'thispersondoesnotexist','path':'https://thispersondoesnotexist.com/image'})
        result = faceIdentification.AddFaceIDs('unitestModel.pkl',objectsToTrainList)
        frame = face_recognition.load_image_file('./MaskIDTest/2161_Mask.jpg')
        face_encoding = face_recognition.face_encodings(frame)[0]
        result = faceIdentification.IdentifyFace(face_encoding)
        expected = {'status': True, 'comment': 'face image was found on ST_FaceID.Model', 'faceID': 'UrlArturoGaona'}
        self.assertEqual(result,expected)

if __name__ == '__main__':
    unittest.main()

######################################################################################
#  File Revision History (top to bottom: first revision to last revision)
#
#
# Date          userid          Description                                   
# 26-Aug-2020   Arturo Gaona    first release of the design implementation    
# 20-Sep-2020   Arturo Gaona    -Include capability to train images from url  
#                               -Implement solution for issues: 
#                                   * DBL_40
#                                   * DBL_24
#                                   * DBL_21
#                                   * DBL_41
#                                       
#########################################################################################