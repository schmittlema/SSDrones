
# coding: utf-8

# In[7]:

import numpy as np
import tensorflow as tf
import time
import os.path

from __future__ import print_function


# In[8]:

import math

class Layer(object):
    def __init__(self, input_sizes, output_size):
        """Cretes a neural network layer."""
        if type(input_sizes) != list:
            input_sizes = [input_sizes]
        
        self.input_sizes = input_sizes
        self.output_size = output_size
                
        self.Ws = []
        for input_size in input_sizes:
            tensor_W = tf.random_uniform((input_size, output_size),
                                         -1.0 / math.sqrt(input_size),
                                         1.0 / math.sqrt(input_size))
            self.Ws.append(tf.Variable(tensor_W))

        tensor_b = tf.zeros((output_size,))
        self.b = tf.Variable(tensor_b)
            
    def __call__(self, xs):
        if type(xs) != list:
            xs = [xs]
        assert len(xs) == len(self.Ws),                 "Expected %d input vectors, got %d" % (len(self.Ws), len(x))
        return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

        
class MLP(object):
    def __init__(self, input_sizes, hiddens, nonlinearities,brainName):
        self.input_sizes = input_sizes
        self.hiddens = hiddens
        self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
        self.count = 0
        self.bN = brainName
        assert len(hiddens) == len(nonlinearities),                 "Number of hiddens must be equal to number of nonlinearities"
        
        self.input_layer = Layer(input_sizes, hiddens[0])
        self.layers = [Layer(h_from, h_to) for h_from, h_to in zip(hiddens[:-1], hiddens[1:])]

    def __call__(self, xs):
        self.count = self.count +1 
        if(self.count == 10):
            self.count = 0
        self.saveNetwork()
        if type(xs) != list:
            xs = [xs]
        hidden = self.input_nonlinearity(self.input_layer(xs))
        for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
            hidden = nonlinearity(layer(hidden))
        return hidden
    
    def saveNetwork(self):
        path = "../saved_brains/"
        name = "Brain:_"# + self.bN
        name = os.path.join(path,name)
        wtf = open(name,"w")
        #wtf.write(self.input_layer)
        wtf.write("hi")
        wtf.close()


# In[9]:

#Test code
#mlp = MLP(5,[5,5,5],[5,5,5],"test")
#for i in range(0,10):
 #   mlp(5.0)

