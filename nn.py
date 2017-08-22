import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#Input and Output 

X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
y=np.array([[1],[1],[0]])

#Sigmoid function

def sigmoid (x):
    return 1/(1+np.exp(-x))

def derivative_sigmoid (x):
    return x*(1-x)

#Setting training iterations
epochs=5000 
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] 
hiddenlayer_neurons = 4 
output_neurons = 1 


#weight and bias initialization

wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


for i in range(epochs):
    
    hidden_input=np.dot(X,wh)
    hidden_activation=sigmoid((hidden_input +bh))
    output_layer_input=np.dot(hidden_activation,wout)
    final_output=sigmoid ((output_layer_input)+bout)
    
    E= y-final_output
    slope_output_layer = derivative_sigmoid(final_output)
    
    slope_hidden_layer = derivative_sigmoid(hidden_activation)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hidden_activation.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    
print (final_output)


    
