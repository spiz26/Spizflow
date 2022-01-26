import numpy as np
import pandas as pd

def sigmoid(x):
    """Sigmoid function"""
    return 1/(1 + np.exp(-x))
    
def d_sigmoid(x):
    """Sigmoid derivative"""
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def relu(x):
    """ReLU function"""
    return np.where(x < 0, 0, x)

def d_relu(x):
    """ReLU derivative"""
    return np.where(x < 0, 0, 1)

def softmax(x):
    """Softmax function"""
    x = x - np.max(x, axis=1).reshape(len(x),1)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims = True)

def OneHot(label_data, num_class):
    return np.identity(num_class)[label_data]