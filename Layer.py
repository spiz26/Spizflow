import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as stamp
from abc import ABC, abstractmethod

from Activation import *
from Optimizer import *
from spizflow_functions import *

class Layer(ABC):
    """Layer Abstract Class"""
    @abstractmethod
    def __init__(self):
        self.type = None
        
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backprop(self):
        pass
    
    def update(self, alpha):
        pass
    
class Dense(Layer):
    """Dense Layer"""
    def __init__(self, pre_neurons, neurons, activation='relu',
                 optimizer='Adam', name=None):
        self.name = name
        self.type = ('layer','dense')
        
        self.activation_dict = {'linear' : Linear(),
                                'sigmoid' : Sigmoid(),
                                'tanh' : tanh(),
                                'softmax' : Softmax(),
                                'relu' : ReLU(),
                                'leakey_relu' : Leaky_ReLU(),
                                'elu' : ELU(),
                                'prelu' : PReLU()}
        
        self.optimizer_dict = {'SGD' : SGD(),
                               'Momentum' : Momentum(),
                               'Adagrad' : AdaGrad(),
                               'RMSprop' : RMSprop(),
                               'Adam' : Adam()}
        
        self.activation = self.activation_dict[activation]
        self.optimizer = self.optimizer_dict[optimizer]
        
        self.W, self.b = Initialization(self.type, activation, (pre_neurons, neurons))
        self.params = {'Weight' : self.W, 'bias' : self.b}
        self.grads = {}
        
    def forward(self, x):
        self.x = x
        self.z = self.x @ self.W + self.b
        self.y = self.activation.forward(self.z)
        
        return self.y
    
    def backprop(self, dy):
        delta = self.activation.backprop(dy)

        self.dW = self.x.T @ delta
        self.db = np.sum(delta, axis=0)    
        self.dx = delta @ self.W.T
        
        self.grads = {'Weight' : self.dW, 'bias' : self.db}
        
        return self.dx
    
    def update(self, alpha):
        self.optimizer.update(self.params, self.grads, alpha)
        
    def __str__(self):
        """Print weight and bias"""
        return (f"{self.name}'s Layer W\n{self.W}\n\n{self.name}'s Layer b\n{self.b}")
    
class Dropout(Layer):
    """Dropout layer"""
    def __init__(self, dropout_ratio, name=None):
        self.name = name
        self.type = ('layer','dropout')
        self.dropout_ratio = dropout_ratio
        
    def forward(self, x, is_train):
        if is_train:
            rand = np.random.rand(*x.shape)
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)
            self.y = x * self.dropout
        else:
            self.y = (1-self.dropout_ratio)*x
        return self.y
        
    def backprop(self, dy):
        self.dx = dy * self.dropout
        return self.dx
    
    def __str__(self):
        """Print Layer feature"""
        return (f"{self.name}'s dropout ratio is {self.dropout_ratio}")
    
class FCLayer(Layer):
    """Fully Connected Layer"""
    def __init__(self, name=None):
        self.name = name
        self.type = ('layer', 'fc')
        
    def forward(self, x):
        self.x = x
        self.num = self.x.shape[0]
        self.y = x.reshape(self.num, -1)
        return self.y
    
    def backprop(self, dy):
        self.dx = dy.reshape(*self.x.shape)
        return self.dx

class Conv2D(Layer):
    """Convolution Layer"""
    def __init__(self, x_shape, flt_shape, padding=0, stride=1, activation='relu',
                 optimizer='Adagrad', name=None):
        self.name = name
        self.type = ('layer','conv2d')

        self.activation_dict = {'linear' : Linear(),
                                'sigmoid' : Sigmoid(),
                                'tanh' : tanh(),
                                'softmax' : Softmax(),
                                'relu' : ReLU(),
                                'leakey_relu' : Leaky_ReLU(),
                                'elu' : ELU(),
                                'prelu' : PReLU()}
        
        self.optimizer_dict = {'SGD' : SGD(),
                               'Momentum' : Momentum(),
                               'Adagrad' : AdaGrad(),
                               'RMSprop' : RMSprop(),
                               'Adam' : Adam()}

        self.activation = self.activation_dict[activation]
        self.optimizer = self.optimizer_dict[optimizer]

        self.x_shape = x_shape
        self.x_ch, self.x_h, self.x_w = x_shape
        self.num_flt, self.flt_h, self.flt_w = flt_shape

        self.W, self.b = Initialization(self.type, activation,
                                       (self.num_flt, self.x_ch, self.flt_h, self.flt_w))
                
        self.params = {'Weight' : self.W, 'bias' : self.b}
        self.grads = {}
        
        self.stride = stride
        self.padding = padding
        
        self.y_ch = self.num_flt 
        self.y_h = (self.x_h - self.flt_h + 2*self.padding) // self.stride + 1
        self.y_w = (self.x_w - self.flt_w + 2*self.padding) // self.stride + 1
        
    def forward(self, x):
        self.num_batch = x.shape[0]
        
        if x.ndim != 4:
            x = x.reshape(self.num_batch, self.x_ch, self.x_h, self.x_w)
        
        self.cols = im2col(x, self.flt_h, self.flt_w, self.padding, self.stride)
        self.W_col = self.W.reshape(self.num_flt, self.x_ch*self.flt_h*self.flt_w)
        
        self.z = (self.W_col @ self.cols).T + self.b
        self.z = self.z.reshape(self.num_batch, self.y_h, self.y_w, self.y_ch)
        self.z = self.z.transpose(0, 3, 1, 2)
        
        self.y = self.activation.forward(self.z)
        return self.y
    
    def backprop(self, dy):
        delta = self.activation.backprop(dy)
        delta = delta.transpose(0,2,3,1).reshape(self.num_batch*self.y_h*self.y_w, self.y_ch)
        
        dW = self.cols @ delta
        self.dW = dW.T.reshape(self.num_flt, self.x_ch, self.flt_h, self.flt_w)
        self.db = np.sum(delta, axis=0)
        
        dcols = delta @ self.W_col
        x_shape = (self.num_batch, *self.x_shape)
        self.dx = col2im(dcols.T, x_shape, self.flt_h, self.flt_w, self.padding, self.stride)
        
        self.grads = {'Weight' : self.dW, 'bias' : self.db}
        return self.dx
    
    def update(self, alpha):
        self.optimizer.update(self.params, self.grads, alpha)
        
    def y_shape(self):
        self.y_shape = (self.y_ch, self.y_h, self.y_w)
        return self.y_shape
    
class Pooling(Layer):
    """Pooling Layer"""
    def __init__(self, x_shape, pool, padding=0, name=None):
        self.name = name
        self.type = ('layer','pool')
        
        self.x_shape = x_shape
        self.x_ch, self.x_h, self.x_w = x_shape
        self.pool = pool
        self.padding = padding

        self.y_ch = self.x_ch
        self.y_h = self.x_h // pool if self.x_h % pool==0 else self.x_h // pool+1
        self.y_w = self.x_w // pool if self.x_w % pool==0 else self.x_w // pool+1

    def forward(self, x):
        self.num_batch = x.shape[0]
        
        cols = im2col(x, self.pool, self.pool, self.padding, self.pool)
        cols = cols.T.reshape(self.num_batch*self.y_h*self.y_w*self.x_ch, self.pool*self.pool)
        
        y = np.max(cols, axis=1)
        self.y = y.reshape(self.num_batch, self.y_h, self.y_w, self.x_ch).transpose(0, 3, 1, 2)
        
        self.max_index = np.argmax(cols, axis=1)
        return self.y
    
    def backprop(self, dy):
        dy = dy.transpose(0, 2, 3, 1)
        
        dcols = np.zeros((self.pool*self.pool, dy.size))
        dcols[self.max_index.reshape(-1), np.arange(dy.size)] = dy.reshape(-1) 
        dcols = dcols.reshape(self.pool, self.pool, self.num_batch, self.y_h, self.y_w, self.y_ch)
        dcols = dcols.transpose(5,0,1,2,3,4) 
        dcols = dcols.reshape(self.y_ch*self.pool*self.pool, self.num_batch*self.y_h*self.y_w)

        x_shape = (self.num_batch, self.x_ch, self.x_h, self.x_w)
        self.dx = col2im(dcols, x_shape, self.pool, self.pool, self.padding, self.pool)
        return self.dx
    
    def y_shape_2D(self):
        self.y_shape = (self.y_ch, self.y_h, self.y_w)
        return self.y_shape
    
    def y_shape_fc(self):
        l = 1
        for i in self.y_shape_2D():
            l *= i
        return l

class Model:
    """Model Declaration"""
    def __init__(self, name=None):
        self.name = name
        self.LayerList = []
        self.train_loss = []
        self.test_loss = []
        
    def add(self, layer):
        """Layer adding method"""
        self.LayerList.append(layer)
    
    def _forward(self, x_data, is_train):
        """Forward propagation method"""
        for layer in self.LayerList:
            if layer.type[1] == 'dropout':
                x_data = layer.forward(x_data, is_train)
            else:
                x_data = layer.forward(x_data)
        return x_data
    
    def _backprop(self, y_data):
        """Backward propagation method"""
        for layer in self.LayerList[::-1]:
            y_data = layer.backprop(y_data)
        return y_data
    
    def _update(self, alpha):
        """Parameters update method"""
        for layer in self.LayerList:
            layer.update(alpha)
            
    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs, alpha=0.01):
        """Fitting function"""
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        num_batch = num_train // batch_size
        
        self._forward(X_train, False)
        print(f"[0/{epochs} epochs] ",end='')
        self.score(X_train, y_train, X_test, y_test)
        
        t0 = stamp()
        for epoch in range(epochs):
            rand_idx = np.arange(num_train)
            np.random.shuffle(rand_idx)
            for mini_batch in range(num_batch):
                mb_index = rand_idx[mini_batch*batch_size:(mini_batch + 1)*batch_size]
                x = X_train[mb_index, :]
                y = y_train[mb_index, :]
                
                self._forward(x, True)
                self._backprop(y)
                self._update(alpha)
            
            print(f"[{epoch+1}/{epochs} epochs] ",end='')
            self.score(X_train, y_train, X_test, y_test)
            
        t1 = stamp()
        total_time = t1-t0
        minute = total_time // 60
        second = round(total_time % 60, 2)
        
        print(f"Training complete! {minute}minutes, {second}seconds")
        
    def score(self, X_train, y_train, X_test, y_test):
        """Score method"""
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        
        predict_y1 = self._forward(X_train, False)
        count_train = np.sum(np.argmax(predict_y1, axis=1) == np.argmax(y_train, axis=1))
        self.train_loss.append(Loss_Cross_Entropy(predict_y1, y_train, num_train))
        
        predict_y2 = self._forward(X_test, False)
        count_test = np.sum(np.argmax(predict_y2, axis=1) == np.argmax(y_test, axis=1))
        self.test_loss.append(Loss_Cross_Entropy(predict_y2, y_test, num_test))
        
        print(f"Train Accuracy : {round(count_train / num_train*100,3)}%,",
              f"Test Accuracy : {round(count_test / num_test*100,3)}%")
        
    def predict(self, x):
        """Predict method"""
        return self._forward(x, False)
    
    def total_parameters(self):
        self.total_parameters = 0
        for layer in self.LayerList:
            if layer.type != 'Drop':
                self.total_parameters += layer.W.size + layer.b.size
        return self.total_parameters
    
    def print_loss(self):
        fig = plt.figure(figsize=(12,5))
        
        ax1 = fig.add_subplot(121)
        x = np.arange(0, len(self.train_loss))
        y_train = self.train_loss
        ax1.set_title("Train set loss", size = 15)
        plt.xlabel("epochs")
        plt.plot(x, y_train)
        
        ax2 = fig.add_subplot(122)
        y_test = self.test_loss
        ax2.set_title("Test set loss", size = 15)
        plt.xlabel("epochs")
        plt.plot(x, y_test)
        
        print(f"Train Loss : {round(self.train_loss[-1],3)},",
              f"Test Loss : {round(self.test_loss[-1],3)}")