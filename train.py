# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:39:55 2017

@author: bebxadvaboy
"""
from helper import create_data_set
from network import FaceDetector
def train():
    
    X_train, X_test, y_train, y_test  = create_data_set() 
    fd = FaceDetector(epochs=50, batch_size=X_train.shape[1], learning_rate = .001)
   
    fd.fit(X_train, y_train, X_test,  y_test )
    return 0


if __name__ == '__main__':
    
    train()