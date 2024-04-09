# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:42:29 2024

@author: ganClimb
"""

import numpy as np
import math


class Type(object):
    @classmethod
    def str(cls):
        raise NotImplementedError


class InitType(Type):
    """Standard names for init
    """
    UNIFORM = 'uniform'
    NORMAL = "normal"
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'

    def str(self):
        return ",".join(
            [self.UNIFORM, self.NORMAL, self.XAVIER_UNIFORM, self.XAVIER_NORMAL,
             self.KAIMING_UNIFORM, self.KAIMING_NORMAL])
    
    
def _xavier_uniform(input_dim,output_dim):
    assert type(input_dim) == int and type(output_dim) == int
    v = math.sqrt(6.0/(input_dim + output_dim))
    return np.random.uniform(-v,v,size = (input_dim,output_dim))

def _xavier_normal(input_dim,output_dim):
    assert type(input_dim) == int and type(output_dim) == int
    std = math.sqrt(2.0/(input_dim + output_dim))
    return np.random.normal(loc = 0,scale = std,size = (input_dim,output_dim))
    
def _kaiming_uniform(input_dim,output_dim):
    assert type(input_dim) == int and type(output_dim) == int
    v = math.sqrt(6.0/input_dim)
    return np.random.uniform(-v,v,size = (input_dim,output_dim))

def _kaiming_normal(input_dim,output_dim):
    assert type(input_dim) == int and type(output_dim) == int
    std = math.sqrt(2.0/input_dim)
    return np.random.normal(loc = 0,scale = std,size = (input_dim,output_dim))
    
    
    
    
    
    
    
    