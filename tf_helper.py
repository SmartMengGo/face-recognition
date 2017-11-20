# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:53:59 2017

@author: bebxadvaboy
"""

import tensorflow as tf

class Layers(object):
    def __init__(self):
        pass
    def init_weights(self,shape):
        
        init_random_dist =tf.random_normal(shape, mean=0, stddev=0.03)
        
        return tf.Variable(init_random_dist)


    def init_bias(self,shape):
        
        init_bias_vals = tf.constant(0.0, shape=shape)
        
        return tf.Variable(init_bias_vals)
    
    def normal_full_layer(self,input_layer, size, layer_name="normal_layer"):
        
        with tf.name_scope(layer_name):
    
        
            input_size = int(input_layer.get_shape()[1])
            
            with tf.name_scope(layer_name+'_weights'):
            
                W = self.init_weights([input_size, size])
            
            with tf.name_scope(layer_name+'_biases'):
    
                b = self.init_bias([size])
    
        return tf.matmul(input_layer, W) + b