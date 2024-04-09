# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:47:34 2024

@author: ganClimb
"""

import numpy as np
from nn_utils import (InitType,
                         _xavier_normal,
                         _xavier_uniform,
                         _kaiming_normal,
                         _kaiming_uniform)


# 单层全连接层
class SingleLinearLayer(object):
    def __init__(self,input_dim,output_dim):
        '''
        @param input_dim:输入维度
               output_dim:输出维度
        '''
        self.input_dim = input_dim
        self.output_dim = output_dim
        print('\tFully connected layer with input %d, output %d.' % (self.input_dim, self.output_dim))
        
        self.weight = np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim, self.output_dim))
        self.bias = np.zeros([1, self.output_dim])
        
    def _init_param(self,init_type = 'uniform'):
        if init_type == 'uniform':
            self.weight = np.random.uniform(0,1,size = (self.input_dim,self.output_dim))
        elif init_type == 'normal':
            self.weight = np.random.normal(loc=0.0, scale=0.01, size=(self.input_dim, self.output_dim))
        elif init_type == 'xavier_uniform':
            self.weight == _xavier_uniform(self.input_dim,self.output_dim)
        elif init_type == 'xavier_normal':
            self.weight == _xavier_normal(self.input_dim,self.output_dim)
        elif init_type == 'kaiming_uniform':
            self.weight == _kaiming_uniform(self.input_dim,self.output_dim)
        elif init_type == 'kaiming_normal':
            self.weight == _kaiming_normal(self.input_dim,self.output_dim)
        else:
            raise TypeError("Unsupported tensor init type: %s. Supported init type is: %s" % (
                init_type, InitType.str()))
            
            
    def forward(self,inputs):
        inputs_dim = inputs.shape[-1]
        assert inputs_dim == self.input_dim
        self.inputs = inputs
        outputs = np.matmul(inputs,self.weight) + self.bias
        return outputs
    
    
    def backward(self,delta):
        self.delta_w = np.dot(self.weight.T,delta)
        self.delta_b = np.sum(delta , axis = 0)
        delta_backword = np.dot(delta, self.weight.T)
        return delta_backword
    
    def update_parameters(self,learning_rate):
        self.weight = self.weight - learning_rate*self.delta_w
        self.bias = self.bias - learning_rate*self.delta_b
        
    
    def load_param(self,weight,bias):
        assert weight.shape == self.weight.shape
        assert bias.shape == self.bias.shape
        self.weight = weight
        self.bias = bias
        
    def get_param(self):
        return self.weight,self.bias
    

# 深层全连接层    
class MultiLinearLayer(object):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.input_layer = SingleLinearLayer(self.input_dim, self.hidden_dim)
        self.output_layer = SingleLinearLayer(self.hidden_dim, self.output_dim)
        hidden_layer = [SingleLinearLayer(self.hidden_dim, self.hidden_dim) for
                               _ in range(self.num_layers-1)]
        self.all_layers = [self.input_layer] + hidden_layer + [self.output_layer]
        self.delta_w = []
        self.delta_b = []
        self.delta_backward = []
        
    def _init_param(self,init_type = 'uniform'):
        for module in self.all_layers:
            module._init_param(init_type)
            
            
    def forward(self,inputs):
        inputs_dim = inputs.shape[-1]
        assert inputs_dim == self.input_dim
        outputs = inputs
        for module in self.all_layers:
            outputs = module.forward(outputs)
        return outputs
    
    def backward(self,delta):
        #delta_beckward = self.output_layer.beckward(delta)
        for i in range(self.num_layers,-1,-1):
            delta = self.all_layers[i].backward(delta)
        return delta
        
    def update_parameters(self,learning_rate):
        if isinstance(learning_rate, list):
            print('Training with different lr !')
            for i in range(self.num_layers+1):
                module = self.all_layers[i]
                module.weight = module.weight - learning_rate[i]*module.delta_w
                module.bias = module.bias - learning_rate[i]*module.delta_b
        elif type(learning_rate) == float:
            for module in self.all_layers:
                module.weight = module.weight - learning_rate*module.delta_w
                module.bias = module.bias - learning_rate*module.delta_b
        else:
            raise TypeError('Type of learning_rate must be float or list !')
            
    def load_param(self,weight,bias):
        '''
        @param weight-->list[array1,array2] len of num_layers+1
               bias-->list
        '''
        for i in range(self.num_layers+1):
            module = self.all_layers[i]
            assert weight.shape == module.weight.shape
            assert bias.shape == module.bias.shape
            module.weight = weight
            module.bias = bias
        
    def get_param(self):
        self.weight = []
        self.bias = []
        for module in self.all_layers:
            self.weight.append(module.weight)
            self.bias.append(module.bias)
        return self.weight,self.bias
    
    
    