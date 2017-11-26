# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:40:13 2017

@author: bebxadvaboy
"""



import tensorflow as tf
from datetime import datetime
import os
import numpy as np


class FaceDetector(object):
    
    def __init__(self, dropout=.2, epochs=3, batch_size=5, learning_rate = .00001):
        
        self.layer = Layers()
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
    def neural_net(self,x):
        """
        Multi-layer perceptron.
        Returns a multi-layer-perceptron to use with tensorflow
        
        Positional arguments:
            
            x -- tensorflow place holder for input data
        
        """

        # Hidden fully connected layer with 512 neurons
        layer_1 = tf.layers.dense(x, 512)
        # Hidden fully connected layer with 512 neurons
        layer_2 = tf.layers.dense(layer_1, 512)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, self.num_classes)
       
        return out_layer
    
    def build_model(self, input_size, output_size):
        
        """
        Build a tensorflow model  for multi-label classification
        
        Positional arguments:
            
            input_size -- dimension of the input samples
            
            output_size -- dimension of the labels 
            
        
        """
        input_x = tf.placeholder(tf.float32, [None, input_size], name="input_x")
        input_y = tf.placeholder(tf.int32, [None, output_size], name="input_y")
        

        y_pred= self.neural_net(input_x)
        
        with tf.name_scope('cross_entropy'):
            
             cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=y_pred))
             
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(cross_entropy)
        
        

        return train, cross_entropy, input_x, input_y, y_pred
    
    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Fit a tensorflow model
        
        Positional arguments:
            
            X_train -- numpy ndarray of training input
            
            y_train -- one-hot-encoded training labels 
            
        Keyword arguments:
            
            X_valid -- numpy ndarray of validation input
            
            y_valid -- one-hot-encoded validation labels 
            
        
        
        """
        self.num_classes = y_train.shape[1]
        
        print(y_train.shape)
        train, cross_entropy, input_x, input_y , y_pred= self.build_model(X_train.shape[1], self.num_classes )
        init = tf.global_variables_initializer()
        steps = X_train.shape[0]
        with tf.Session() as sess:
            
            
            saver = tf.train.Saver()
            sess.run(init)
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                print("")
                shape = variable.get_shape()
                print(shape)
                print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    print(dim)
                    variable_parameters *= dim.value
                print(variable_parameters)
                total_parameters += variable_parameters
                print("")
            print("total_parameters",total_parameters)
            while True:
                for i in range(0, steps, self.batch_size):
                    
                    x_batch_train = X_train[i: i+self.batch_size]
                    y_batch_train = y_train[i: i+self.batch_size]
                    
                    tr, ce, pr =sess.run([train,cross_entropy, y_pred],feed_dict={input_x:x_batch_train, input_y:y_batch_train})

                    print("{} iterations: {}   loss: {}  ".format(str(datetime.now()),i, ce))
                
                if X_valid is not None:
                    
                    print("\n\nEvaluation...")
                    tr, ce =sess.run([train,cross_entropy],feed_dict={input_x:X_valid, input_y:y_valid})
                
                    print("{} iterations: {}   loss: {}  ".format(str(datetime.now()),i, ce))                
                    print("\n")
                self.epochs -= 1
                if self.epochs ==0:
                    
                    if not os.path.exists(os.path.join(os.getcwd(), 'saved_model')):
                        os.makedirs(os.path.join(os.getcwd(), 'saved_model'))
                    saver.save(sess, os.path.join(os.getcwd(), 'saved_model','my_test_model'))
                    

                    break
        
    def predict(self, X_valid, y_valid):
        """
        Returns a dictionnary containing the model's predictino as well as the ground truth,
        encoded as integers
        
        Positional arguments:
            
            X_valid -- the validation samples, as numpy flat ndarray
            
            y_valid -- the validation labels, as one-hot-encoded arrays
        
        """
        sess=tf.Session()   
        saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'saved_model','my_test_model.meta'))
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'saved_model')))
        
        
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        y_pred = graph.get_operation_by_name("predictions").outputs[0]
        
        
                
        print("\nMaking predictions...:")
        
        predi_list = []
        ground_truth_list = []
       
           
        print("running sess")
        predi =sess.run(y_pred, feed_dict={input_x:X_valid})
        predi_list += list(predi)
        ground_truth_list +=  [np.argmax(item) for item in y_valid]
    
        pred_dic ={}
        ground_truth = [np.argmax(item) for item in y_valid]
        pred_dic = {'predictions':list(predi), 'ground_truth':ground_truth }
        return pred_dic
