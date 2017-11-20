# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:54:28 2017

@author: bebxadvaboy
"""
from network import FaceDetector
from helper import create_data_set

import matplotlib.pyplot as plt
from utilities import plot_confusion_matrix
import numpy as np

def predict():
    
    
    X_train, X_test, y_train, y_test  = create_data_set() 

    fd = FaceDetector()
    
    pred_dic = fd.predict(X_test,y_test)
    
    plt.figure()
    
    plot_confusion_matrix(pred_dic['ground_truth'],pred_dic['predictions'], classes=range(np.max(pred_dic['predictions'])),
                      title='Confusion matrix, without normalization')

    plt.show()
    
    return pred_dic

if __name__ == '__main__':
    
    pred_dic = predict()