import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Callable, Union, Optional
from abc import ABC, abstractmethod
from Utilis import *

class Activation(ABC):
    """Activation function Abstract Class"""
    @abstractmethod
    def __init__(self):
        self.type = None
    
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def backprop(self):
        pass
    
class Linear(Activation):
    """Linear activation function"""
    def __init__(self):
        self.type = ('activation', 'linear')
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = x
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy

class Sigmoid(Activation):
    """Sigmoid activation function"""
    def __init__(self):
        self.type = ('activation', 'sigmoid')
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = 1 / (1 + np.exp(-x))
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * (1 - self.y) * self.y
#         if not self.last:
#             return dy * (1 - self.y) * self.y
        
#         else:
#             batch_size = self.y.shape[0]
#             num_class = self.y.shape[1]

#             soft_dy = np.zeros((batch_size, num_class, num_class)) #jacobian

#             for i in range(batch_size):
#                 soft_dy[i] = np.diag(self.y[i]) - np.outer(self.y[i], self.y[i])
#             return np.einsum("ni, nij -> nj", dy, soft_dy)
#         diff = differential(batch=True)
#         return dy * diff(self.forward, self.y)
    
class tanh(Activation):
    """Hyperbolic tangent activation function"""
    def __init__(self):
        self.type = ('activation', 'tanh')
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * (1 - self.y) * (1 + self.y)

class Softmax(Activation):
    """Softmax activation function"""
    def __init__(self, with_CE=False):
        self.type = ('activation', 'softmax')
        self.with_CE = with_CE
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        if self.x.ndim == 1:
            x = x - np.max(x)
            self.y = np.exp(x) / np.sum(np.exp(x))
            return self.y
        
        elif self.x.ndim == 2:
            x = x - np.max(x, axis=1).reshape(-1,1)
            self.y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1,1)
            return self.y
        
        else:
            raise Exception("Can't calculate")
    
    def backprop(self, dy: np.array) -> np.array:
        if self.with_CE:
            return dy
        
        if self.y.ndim == 1:
            soft_dy = np.diag(s) - np.outer(s, s)
            return np.einsum("i, ij -> j", dy, soft_dy)
            
        elif self.y.ndim == 2:
            batch_size = self.y.shape[0]
            num_class = self.y.shape[1]

            soft_dy = np.zeros((batch_size, num_class, num_class)) #jacobian

            for i in range(batch_size):
                soft_dy[i] = np.diag(self.y[i]) - np.outer(self.y[i], self.y[i])
            return np.einsum("ni, nij -> nj", dy, soft_dy)
    
class ReLU(Activation):
    """Reductified Linear Unit activation function"""
    def __init__(self):
        self.type = ('activation', 'relu')
    
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = np.where(self.x > 0, x, 0)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * np.where(self.x > 0, 1, 0)
    
class Leaky_ReLU(Activation):
    """Leaky ReLU activation function"""
    def __init__(self):
        self.type = ('activation', 'leaky_relu')
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = np.where(self.x > 0, x, 0.01 * x)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * np.where(self.x > 0, 1, 0.01)

class ELU(Activation):
    """Exponential Linear Unit activation function"""
    def __init__(self):
        self.type = ('activation', 'elu')
        self.alpha = 1.0
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = np.where(self.x > 0, x, self.alpha * (np.exp(self.x)-1))
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * np.where(self.x > 0, 1, self.alpha * np.exp(self.x))

class PReLU(Activation):
    """Parametric ReLU activation function"""
    def __init__(self, alpha: float=0.05):
        self.type = ('activation', 'prelu')
        self.alpha = alpha
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.y = np.where(self.x > 0, x, self.alpha * x)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        return dy * np.where(self.x > 0, 1, self.alpha)