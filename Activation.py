import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Activation(ABC):
    """Activation function Abstract Class"""
    @abstractmethod
    def __init__(self):
        self.type = None
    
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backprop(self):
        pass
    
class Linear(Activation):
    """Linear activation function"""
    def __init__(self):
        self.type = ('activation', 'linear')
        
    def forward(self, x):
        self.x = x
        self.y = x
        return self.y
    
    def backprop(self, dy):
        return dy

class Sigmoid(Activation):
    """Sigmoid activation function"""
    def __init__(self):
        self.type = ('activation', 'sigmoid')
        
    def forward(self, x):
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backprop(self, dy):
        return dy * (np.exp(-self.x)) / ((np.exp(-self.x)+1)**2)

class tanh(Activation):
    """Hyperbolic tangent activation function"""
    def __init__(self):
        self.type = ('activation', 'tanh')
        
    def forward(self, x):
        self.x = x
        self.y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.y
    
    def backprop(self, dy):
        return dy * (1 - self.y) * (1 + self.y)

class Softmax(Activation):
    """Softmax activation function"""
    def __init__(self):
        self.type = ('activation', 'softmax')
    
    def forward(self, x):
        if x.ndim == 1:
            x = x - np.max(x)
            self.pred_y = np.exp(x) / np.sum(np.exp(x))
            return self.pred_y
        
        elif x.ndim == 2:
            x = x - np.max(x, axis=1).reshape(-1,1)
            self.pred_y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)
            return self.pred_y
        
        else:
            raise Exception("Can't calculate")
            
    def backprop(self, true_y):
        return self.pred_y - true_y
    
class ReLU(Activation):
    """Reductified Linear Unit activation function"""
    def __init__(self):
        self.type = ('activation', 'relu')
    
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x > 0, x, 0)
        return self.y
    
    def backprop(self, dy):
        return dy * np.where(self.x > 0, 1, 0)
    
class Leaky_ReLU(Activation):
    """Leaky ReLU activation function"""
    def __init__(self):
        self.type = ('activation', 'leaky_relu')
        
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x > 0, x, 0.01 * x)
        return self.y
    
    def backprop(self, dy):
        return dy * np.where(self.x > 0, 1, 0.01)

class ELU(Activation):
    """Exponential Linear Unit activation function"""
    def __init__(self):
        self.type = ('activation', 'elu')
        self.alpha = 1.0
        
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x > 0, x, self.alpha * (np.exp(self.x)-1))
        return self.y
    
    def backprop(self, dy):
        return dy * np.where(self.x > 0, 1, self.alpha * np.exp(self.x))

class PReLU(Activation):
    """Parametric ReLU activation function"""
    def __init__(self, alpha=0.05):
        self.type = ('activation', 'prelu')
        self.alpha = alpha
        
    def forward(self, x):
        self.x = x
        self.y = np.where(self.x > 0, x, self.alpha * x)
        return self.y
    
    def backprop(self, dy):
        return dy * np.where(self.x > 0, 1, self.alpha)