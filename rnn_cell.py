# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:24:33 2024

@author: Administrator
"""

import numpy as np
from nn_utils import (InitType,
                         _xavier_normal,
                         _xavier_uniform,
                         _kaiming_normal,
                         _kaiming_uniform)

#from activate_function import tanh

class RNNCell(object):
    def __init__(self,input_dim,hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_i = np.random.normal(loc=0,sacle = 0.1,size = (input_dim,hidden_dim))
        self.W_h = np.random.normal(loc=0,sacle = 0.1,size = (hidden_dim,hidden_dim))
        self.bias = np.zeros([1,hidden_dim])
        
    def _init_param(self,init_type = 'uniform'):
        if init_type == 'uniform':
            self.W_i = np.random.uniform(0,1,size = (self.input_dim,self.hidden_dim))
            self.W_h = np.random.uniform(0,1,size = (self.input_dim,self.hidden_dim))
        elif init_type == 'normal':
            self.W_i = np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim, self.hidden_dim))
            self.W_h = np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim, self.hidden_dim))
        elif init_type == 'xavier_uniform':
            self.W_i == _xavier_uniform(self.input_dim,self.hidden_dim)
            self.W_h == _xavier_uniform(self.input_dim,self.hidden_dim)
        elif init_type == 'xavier_normal':
            self.W_i == _xavier_normal(self.input_dim,self.hidden_dim)
            self.W_h == _xavier_normal(self.input_dim,self.hidden_dim)
        elif init_type == 'kaiming_uniform':
            self.W_i == _kaiming_uniform(self.input_dim,self.hidden_dim)
            self.W_h == _kaiming_uniform(self.input_dim,self.hidden_dim)
        elif init_type == 'kaiming_normal':
            self.W_i == _kaiming_normal(self.input_dim,self.hidden_dim)
            self.W_h == _kaiming_normal(self.input_dim,self.hidden_dim)
        else:
            raise TypeError("Unsupported tensor init type: %s. Supported init type is: %s" % (
                init_type, InitType.str()))

    def forward(self,inputs,h):
        '''
        @param inputs-->np.array [batch_size,input_dim]
               h -->np.array [bathc_size,hidden_dim]
        '''
        self.inputs = inputs
        self.h = np.tanh(np.matmul(inputs,self.W_i) + np.matmul(h,self.W_h) + self.bias) # [batch_size,hidden_dim]
        return self.h
    
    def backward(self,delta):
        '''
        @param delta-->np.array [batch_size,hidden_dim]
        '''
        delta = np.matmul(delta,(1-np.matmul(self.h.T,self.h))) # [batch_size,hidden_dim]
        self.delta_wi = np.dot(self.inputs.T , delta) # [input_dim,hidden_dim]
        self.delta_wh = np.dot(self.h.T , delta) # [hidden_dim,hidden_dim]
        self.delta_bias = np.sum(delta,axis = 0) # [1,hidden_dim]
        delta_backward = np.dot(delta,self.W_i.T) # [batch_size,input_dim]
        return delta_backward
    
    def update_parameters(self,learning_rate = 0.05,clip_grad = 3):
        '''
        @param learning_rate:学习率
               clip_grad:梯度裁剪阈值
        '''
        self.W_i = self.W_i - learning_rate*np.clip(self.delta_wi,-clip_grad,clip_grad)
        self.W_h = self.W_h - learning_rate*np.clip(self.delta_wh,-clip_grad,clip_grad)
        self.bias = self.bias - learning_rate*self.delta_bias
        
    
    def load_param(self,weight_i,weight_h,bias):
        assert weight_i.shape == self.W_i.shape
        assert weight_h.shape == self.W_h.shape
        assert bias.shape == self.bias.shape
        self.W_i = weight_i
        self.W_h = weight_h
        self.bias = bias
        
    
    def get_param(self):
        return self.W_i,self.W_h,self.bias
    
    
    def setzero(self):
        self.W_i[...] = 0
        self.W_h[...] = 0
        self.bias[...] = 0
            





