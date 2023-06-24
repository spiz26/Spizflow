import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Callable, Union, Optional
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """Optimizer Abstract Class"""
    @abstractmethod
    def __init__(self):
        self.type = None
    
    @abstractmethod
    def update(self, alpha: float) -> None:
        pass
    
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    def __init__(self):
        self.type = ('optimizer', 'SGD')
    
    def update(self, params: Dict, grads: Dict, alpha=0.01) -> None:
        self.alpha = alpha
        for key in params.keys():
            params[key] -= self.alpha * grads[key]
            
class Momentum(Optimizer):
    """Momentum optimizer"""
    def __init__(self, momentum: float=0.9):
        self.type = ('optimizer', 'momentum')
        self.momentum = momentum
        self.v = None
        
    def update(self, params: Dict, grads: Dict, alpha: float=0.01) -> None:
        self.alpha = alpha
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.alpha * grads[key]
            params[key] += self.v[key]
            
class AdaGrad(Optimizer):
    """Adaptive Gradient optimizer"""
    def __init__(self):
        self.type = ('optimizer', 'adagrad')
        self.h = None
    
    def update(self, params: Dict, grads: Dict, alpha: float=0.01) -> None:
        self.alpha = alpha
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.alpha * grads[key] / (np.sqrt(self.h[key]) + 1e-8)
            
class RMSprop(Optimizer):
    """Root Mean Square Propagation optimizer"""
    def __init__(self, rho: float=0.99):
        self.type = ('optimizer', 'RMSprop')
        self.rho = rho
        self.h = None
    
    def update(self, params: Dict, grads: Dict, alpha: float=0.01) -> None:
        self.alpha = alpha
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += self.rho
            self.h[key] += (1 - self.rho) * grads[key] * grads[key]
            params[key] -= self.alpha * grads[key] / (np.sqrt(self.h[key]) + 1e-8)
            
class Adam(Optimizer):
    """Adapive Moment esimation optimizer"""
    def __init__(self, beta1: float=0.9, beta2: float=0.999):
        self.type = ('optimizer', 'adam')
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: Dict, grads: Dict, alpha: float=0.001) -> None:
        self.alpha = alpha
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        alpha_t = self.alpha * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= alpha_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-8)