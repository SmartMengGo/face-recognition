# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:40:55 2017

@author: bebxadvaboy
"""

import cv2
import numpy as np
import os
from skimage import io
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

OPENCV_HAAR_CSC_PATH = "C:\\Users\\bebxadvaboy\\AppData\\Local\\Continuum\\Anaconda3\\pkgs\\opencv-3.3.0-py36_200\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml"

def create_data_set(x_crop= 150, y_crop=150, train_size=.8): 
    """ Load the Yale Faces data set, extract the faces on the images and generate labels for each image.
        
        Returns: Train and validation samples with their labels. The training samples are flattened arrays 
        of size 22500 (150 * 150) , the labels are one-hot-encoded values for each category
    """
    images_path = [ os.path.join("yalefaces", item)  for item in  os.listdir("yalefaces") ]
    image_data = []
    image_labels = []
    
    for i,im_path in enumerate(images_path):
        im = io.imread(im_path,as_grey=True)
#        if( i== 10) or (i==40) or (i==50):
#            io.imshow(im)
#            io.show()
        image_data.append(np.array(im, dtype='uint8'))
        
        
        
        label = int(os.path.split(im_path)[1].split(".")[0].replace("subject", ""))  -1
       
            
        image_labels.append(label)
    faceDetectClassifier = cv2.CascadeClassifier(OPENCV_HAAR_CSC_PATH)
    
    cropped_faces = []
    for im in image_data:
        facePoints = faceDetectClassifier.detectMultiScale(im)
        x,y = facePoints[0][:2]
        cropped = im[y: y + y_crop, x: x + x_crop]
        cropped_faces.append(cropped/255)
        
    X_ = np.array(cropped_faces).astype('float32')
    enc = LabelEncoder()
    y_ = enc.fit_transform(np.array(image_labels))
    y_ = np_utils.to_categorical(y_)
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=train_size, random_state = 22)

    return (X_train).reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2])), (X_test).reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2])), y_train, y_test
    
    
  