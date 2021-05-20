'''
Created on July, 2020
@author: Jonatan Uresti
'''
#########################################################################################
#  COPYRIGHT 2020
#  \verbatim
#                 This software is copyright protected and proprietary to Condumex.
#                 All other rights remain with Condumex.
#  \endverbatim
#  LICENSE
#          Module: ST_FaceID
#          Program: Configuration file for Face ID module
#          Enterprise: Condumex
#          SW Developer: Arturo Gaona
#          FILE DESCRIPTION
#          File: ST_FaceID_cfg.py
#          Project: EDT_AccessCTRL
#          Delivery: FIRST DELIVERY
#########################################################################################
import os

_debugTest = False
FACE_RECOG_TOLERANCE_MASK = 0.47	#Face Recognition Tolerance with face mask     DBL_9, DBL_21, DBL_12
FACE_RECOG_TOLERANCE_NO_MASK = 0.47	#Face Recognition Tolerance with no face mask  DBL_9, DBL_21, DBL_12
UNKNOWN_USERID = '0000_0000'
FACE_IMAGE_FORMAT = ['jpg']


#MODEL String TAGS
MASK_MODEL = os.path.dirname(__file__) + '/Models/mask_recog_ver4.h5'              
CASCADE_MODEL = os.path.dirname(__file__) + '/Models/haarcascade_frontalface_alt2.xml'

NO_MODEL_EXCEPTION = 'Mask Model NOT found: '
MODEL_FACE_LIST = 'faceIDList'
MODEL_FACE_ENCODE_LIST = 'faceEncodeList'


#Exceptions string TAGS
SINGLETONE_MODEL_EXCEPTION = 'This class is a singleton!'

#General status string TAGS
GENERAL_RESULT_STATUS = 'status'
GENERAL_RESULT_COMMNENT = 'comment'
GENERAL_RESULT_EXCEPTION = 'Exception: '
GENERAL_RESULT_USER_ID = 'sap_number' #@TODO: change to string user_id when correct implementation is done in REST server

#########################################################################################
#  File Revision History (top to bottom: first revision to last revision)
#
# 26-Aug-2020   Arturo Gaona
#   + First release of the design implementation  
#
# 20-Sep-2020   Arturo Gaona
#    -Include capability to train images from url
#    -Implement solution for issues: 
#        * DBL_9, DBL_12,DBL_21
#
# Sep-25-2020 Pablo Mejia
#   + DBL_43
#      - Created initial file.
#      - Adapt POST & PATCH to work with current ST_FACE_ID
#
#########################################################################################