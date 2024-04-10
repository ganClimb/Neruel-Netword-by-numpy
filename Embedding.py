# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:18:53 2024

@author: ganClimb
"""

import numpy as np
from nn_utils import (InitType,
                         _xavier_normal,
                         _xavier_uniform,
                         _kaiming_normal,
                         _kaiming_uniform)
from Linear import SingleLinearLayer


class EmbeddingLayer(object):
    def __init__(self,vocab_size,embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = np.random.normal(loc = 0, scale = 0.1 , size = (vocab_size,embedding_dim))
        self.gradient = np.zeros_like(self.weight)
        
        
    def _init_param(self,init_type = 'uniform'):
        if init_type == 'uniform':
            self.weight = np.random.uniform(0,1,size = (self.vocab_size,self.embedding_dim))
        elif init_type == 'normal':
            self.weight = np.random.normal(loc=0.0, scale=0.1, size=(self.vocab_size, self.embedding_dim))
        elif init_type == 'xavier_uniform':
            self.weight == _xavier_uniform(self.vocab_size,self.embedding_dim)
        elif init_type == 'xavier_normal':
            self.weight == _xavier_normal(self.vocab_size,self.embedding_dim)
        elif init_type == 'kaiming_uniform':
            self.weight == _kaiming_uniform(self.vocab_size,self.embedding_dim)
        elif init_type == 'kaiming_normal':
            self.weight == _kaiming_normal(self.vocab_size,self.embedding_dim)
        else:
            raise TypeError("Unsupported tensor init type: %s. Supported init type is: %s" % (
                init_type, InitType.str()))
            
    def forward(self,inputs):
        '''
        @parama inputs-->np.array [batch_size,seq_len]
        '''
        inputs_size = inputs.shape
        outputs_size = list(inputs_size)+[self.embedding_dim]
        self.flatten_index = inputs.flatten() # [batch_size*seq_len]
        outputs = self.weight[self.flatten_index,:]
        outputs = np.reshape(outputs,outputs_size) # [batch_szie,seq_len,embedding_dim]
        return outputs 
        
    def backward(self,delta):
        # 和全连接层不相同，其他算子一般是全部update，但是Embedding层仅仅需要udpate前向传播使用过的词向量
        '''
        @param delta-->np.array [batch_size,seq_len,embedding_dim]
        '''
        delta_size = delta.shape
        flatten_delta = np.reshape(delta,(delta_size[0]*delta_size[1],delta_size[0]))
        self.gradient[self.flatten_index,:] += flatten_delta
        delta_beckward = self.gradient
        return delta_beckward
        
        
    def update_parameters(self,learning_rate):
        self.weight = self.weight - learning_rate*self.gradient
        
    def _setzero(self):
        self.gradient[...] = 0
        
    def load_param(self,embedding):
        assert embedding.shape == self.weight.shape
        self.weight = embedding
        
    def get_param(self):
        return self.weight
        
        
