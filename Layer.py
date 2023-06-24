import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Set, Callable, Union, Optional
from abc import ABC, abstractmethod
from time import time as stamp

from Activation import *
from Optimizer import *
from Utilis import *

class Layer(ABC):
    """Layer Abstract Class"""
    @abstractmethod
    def __init__(self):
        self.type: Tuple[str, str, Optional[str]] = None
        
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def backprop(self):
        pass
    
    def update(self, alpha: float):
        pass
    
class Dense(Layer):
    """Dense Layer"""
    def __init__(self, pre_neurons: int, neurons: int, activation: Union[str, Activation]='relu',
                 optimizer: Union[str, Optimizer]='Adam', name: Optional[str]=None):
        self.name = name
        self.neurons = neurons
        self.type = ('layer', 'dense', activation)
        
        #activation function
        if isinstance(activation, str):
            self.activation = eval(activation_dict[activation])
                
        elif isinstance(activation, Activation):
            self.activation = activation
            activation = self.activation.type[1]
            
        else:
            raise Exception("Unvalid activation function")
        
        #optimizer
        if isinstance(optimizer, str):
            self.optimizer = eval(optimizer_dict[optimizer])
        
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        
        else:
            raise Exception("Unvalid optimizer")
        
        #parameters & gradient
        self.W, self.b = Initialization(self.type, activation, (pre_neurons, neurons))
        self.params = {'Weight' : self.W, 'bias' : self.b}
        self.grads = {}
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.z = self.x @ self.W + self.b
        self.y = self.activation(self.z)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        delta = self.activation.backprop(dy)

        self.dW = self.x.T @ delta
        self.db = np.sum(delta, axis=0)    
        self.dx = delta @ self.W.T
        
        self.grads = {'Weight' : self.dW, 'bias' : self.db}
        return self.dx
    
    def update(self, alpha: float) -> None:
        self.optimizer.update(self.params, self.grads, alpha)
        
    def __str__(self):
        """Print weight and bias"""
        return (f"Num of layer's nodes is {self.neurons}, layer type is {self.type}")
    
class Dropout(Layer):
    """Dropout layer"""
    def __init__(self, dropout_ratio: float, name: Optional[str]=None):
        self.name = name
        self.type = ('layer', 'dropout', None)
        self.dropout_ratio = dropout_ratio
        
    def __call__(self, x: np.array, is_train: bool) -> np.array:
        if is_train:
            rand = np.random.rand(*x.shape)
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)
            self.y = x * self.dropout
        else:
            self.y = (1-self.dropout_ratio)*x
        return self.y
        
    def backprop(self, dy: np.array) -> np.array:
        self.dx = dy * self.dropout
        return self.dx
    
    def __str__(self):
        """Print Layer feature"""
        return (f"{self.name}'s dropout ratio is {self.dropout_ratio}")
    
class FCLayer(Layer):
    """Fully Connected Layer"""
    def __init__(self, name: Optional[str]=None):
        self.name = name
        self.type = ('layer', 'fc', None)
        
    def __call__(self, x: np.array) -> np.array:
        self.x = x
        self.num = self.x.shape[0]
        self.y = x.reshape(self.num, -1)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        self.dx = dy.reshape(*self.x.shape)
        return self.dx

class Conv2D(Layer):
    """Convolution Layer"""
    def __init__(self, x_shape: Union[List, Tuple], flt_shape: Union[List, Tuple], padding: int=0, stride: int=1, 
                 activation: Union[str, Activation]='relu', optimizer: Union[str, Optimizer]='Adam', name: Optional[str]=None):
        self.name = name
        self.type = ('layer', 'conv2d', activation)

        #activation function
        if isinstance(activation, str):
            self.activation = eval(activation_dict[activation])
                
        elif isinstance(activation, Activation):
            self.activation = activation
            activation = self.activation.type[1]
        
        else:
            raise Exception("Unvalid activation function")
        
        #optimizer
        if isinstance(optimizer, str):
            self.optimizer = eval(optimizer_dict[optimizer])
        
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        
        else:
            raise Exception("Unvalid optimizer")
        
        #parameters & gradient
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
        
    def __call__(self, x: np.array) -> np.array:
        self.num_batch = x.shape[0]
        
        if x.ndim != 4:
            x = x.reshape(self.num_batch, self.x_ch, self.x_h, self.x_w)
        
        self.cols = im2col(x, self.flt_h, self.flt_w, self.padding, self.stride)
        self.W_col = self.W.reshape(self.num_flt, self.x_ch*self.flt_h*self.flt_w)
        
        self.z = (self.W_col @ self.cols).T + self.b
        self.z = self.z.reshape(self.num_batch, self.y_h, self.y_w, self.y_ch)
        self.z = self.z.transpose(0, 3, 1, 2)
        
        self.y = self.activation(self.z)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
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
    
    def update(self, alpha: float) -> None:
        self.optimizer.update(self.params, self.grads, alpha)
        
    def y_shape(self) -> Tuple:
        self.y_shape = (self.y_ch, self.y_h, self.y_w)
        return self.y_shape
    
class Pooling(Layer):
    """Pooling Layer"""
    def __init__(self, x_shape: Union[List, Tuple], pool: int, padding: int=0, name: Optional[str]=None):
        self.name = name
        self.type = ('layer', 'pool', None)
        
        self.x_shape = x_shape
        self.x_ch, self.x_h, self.x_w = x_shape
        self.pool = pool
        self.padding = padding

        self.y_ch = self.x_ch
        self.y_h = self.x_h // pool if self.x_h % pool==0 else self.x_h // pool + 1
        self.y_w = self.x_w // pool if self.x_w % pool==0 else self.x_w // pool + 1

    def __call__(self, x: np.array) -> np.array:
        self.num_batch = x.shape[0]
        
        cols = im2col(x, self.pool, self.pool, self.padding, self.pool)
        cols = cols.T.reshape(self.num_batch*self.y_h*self.y_w*self.x_ch, self.pool*self.pool)
        
        y = np.max(cols, axis=1)
        self.y = y.reshape(self.num_batch, self.y_h, self.y_w, self.x_ch).transpose(0, 3, 1, 2)
        
        self.max_index = np.argmax(cols, axis=1)
        return self.y
    
    def backprop(self, dy: np.array) -> np.array:
        dy = dy.transpose(0, 2, 3, 1)
        
        dcols = np.zeros((self.pool*self.pool, dy.size))
        dcols[self.max_index.reshape(-1), np.arange(dy.size)] = dy.reshape(-1) 
        dcols = dcols.reshape(self.pool, self.pool, self.num_batch, self.y_h, self.y_w, self.y_ch)
        dcols = dcols.transpose(5,0,1,2,3,4) 
        dcols = dcols.reshape(self.y_ch*self.pool*self.pool, self.num_batch*self.y_h*self.y_w)

        x_shape = (self.num_batch, self.x_ch, self.x_h, self.x_w)
        self.dx = col2im(dcols, x_shape, self.pool, self.pool, self.padding, self.pool)
        return self.dx
    
    def y_shape_2D(self) -> Tuple:
        self.y_shape = (self.y_ch, self.y_h, self.y_w)
        return self.y_shape
    
    def y_shape_fc(self) -> int:
        l = 1
        for i in self.y_shape_2D():
            l *= i
        return l

class Loss(Layer):
    def __init__(self, loss_name: str, BuiltIn_derivative: bool, last_layer: Layer):
        self.loss_name = loss_name
        self.type = ("layer", "loss", None)
        self.BuiltIn_derivative = BuiltIn_derivative
        
        if self.loss_name == "Softmax_with_CrossEntropy":
            self.BuiltIn_derivative = True

        if last_layer.type[2] == "softmax":
            self.diff = differential(iterdim=0, batch=True)
        else:
            self.diff = differential(iterdim=1, batch=True)
        
    def __call__(self, x: np.array) -> np.array:
        self.y = x
        return self.y
    
    def backprop(self, y_data: np.array) -> np.array:
        if self.BuiltIn_derivative:
            self.dx = eval(Loss_dict[self.loss_name]+"_diff(self.y, y_data)")

        else:
            self.dx = self.diff(eval(Loss_dict[self.loss_name]), self.y, y_data)
            if self.dx.ndim == 1:
                self.dx = self.dx.reshape(-1, 1)
        return self.dx
    
class SequentialModel:
    """Model Declaration"""
    def __init__(self, name: Optional[str]=None):
        self.name = name
        self.LayerList = []
        self.train_loss = []
        self.test_loss = []

    def add(self, layer: Layer) -> None:
        """Layer adding method"""
        self.LayerList.append(layer)
    
    def _forward(self, x_data: np.array, is_train: bool) -> np.array:
        """Forward propagation method"""
        layer_type = None
        for layer in self.LayerList:
            if layer.type[1] == 'dropout':
                x_data = layer(x_data, is_train)
            else:
                x_data = layer(x_data)
        return x_data
    
    def _backprop(self, y_data: np.array) -> np.array:
        """Backward propagation method"""
        for layer in self.LayerList[::-1]:
            y_data = layer.backprop(y_data)
        return y_data
    
    def _update(self, alpha: float) -> None:
        """Parameters update method"""
        for layer in self.LayerList:
            layer.update(alpha)
    
    def model_compile(self, batch_size: int, epochs: int, lr: float=0.01, verbose: bool=True,
                      loss: str="mse", built_in_diff: bool=True, metric: str="mse") -> None:
        """model hyper parameters setting"""
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.loss = loss
        self.metric = metric
        self.built_in_diff = built_in_diff
        
        if loss not in list(Loss_dict.keys()):
            raise Exception("Unvalid loss")
        
        if metric not in list(metrics_set):
            raise Exception("Unvalid metric")
            
    def fit(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array) -> Tuple[List, List]:
        """Fitting function"""
        self.LayerList.append(Loss(self.loss, self.built_in_diff, self.LayerList[-1]))
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        num_batch = num_train // self.batch_size
        
        self._forward(X_train, False)
        print(f"[0/{self.epochs} epochs] ",end='\n')
        self.score(X_train, y_train, X_test, y_test)
        
        t0 = stamp()
        for epoch in range(self.epochs):
            rand_idx = np.arange(num_train)
            np.random.shuffle(rand_idx)
            if self.verbose:
                print(f"[{epoch+1}/{self.epochs} epochs]")
            
            for mini_batch in range(num_batch):
                mb_index = rand_idx[mini_batch*self.batch_size:(mini_batch + 1)*self.batch_size]
                x = X_train[mb_index, :]
                y = y_train[mb_index, :]
                
                self._forward(x, True)
                self._backprop(y)
                self._update(self.lr)
                
                if self.verbose:
                    width = process_visualizing(mini_batch, num_batch)
                    
            if self.verbose:
                print(f"\rmini batch : 100%","[" + "="*width + "]" + " "*10)

            history = self.score(X_train, y_train, X_test, y_test)
            
        t1 = stamp()
        total_time = t1-t0
        minute = total_time // 60
        second = round(total_time % 60, 2)
        
        print(f"Training complete! {minute}minutes, {second}seconds")
        return history
        
    def score(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array) -> Tuple[List, List]:
        """Score method"""
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        loss_name = Loss_dict[self.loss]
        
        #inference
        predict_train = self._forward(X_train, False)
        predict_test = self._forward(X_test, False)
        
        if self.metric == "rmse":
            #train set
            self.train_loss.append(Loss_RMSE(predict_train, y_train))
            
            #test set
            self.test_loss.append(Loss_RMSE(predict_test, y_test))
            
            if self.verbose:
                print(f"Train {self.metric} : {self.train_loss[-1]},", 
                      f"Test {self.metric} : {self.test_loss[-1]}", end="\n\n")
            
        #Accuracy
        elif self.metric == "accuracy":
            #train set
            count_train = np.sum(np.argmax(predict_train, axis=1) == np.argmax(y_train, axis=1))
            train_accuracy = round(count_train / num_train*100,3)
            self.train_loss.append(np.mean(Loss_Cross_Entropy(predict_train, y_train)))
            
            #test set
            count_test = np.sum(np.argmax(predict_test, axis=1) == np.argmax(y_test, axis=1))
            test_accuracy = round(count_test / num_test*100,3)
            self.test_loss.append(np.mean(Loss_Cross_Entropy(predict_test, y_test)))
            
            if self.verbose:
                print(f"Train Accuracy : {train_accuracy}%,",
                      f"Test Accuracy : {test_accuracy}%", end="\n\n")
        return self.train_loss, self.test_loss
        
    def __call__(self, x) -> np.array:
        """inference method"""
        return self._forward(x, False)
    
    def total_parameters(self) -> int:
        self.total_parameters = 0
        for layer in self.LayerList:
            if layer.type != 'Drop':
                self.total_parameters += layer.W.size + layer.b.size
        return self.total_parameters
    
    def print_loss(self) -> None:
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