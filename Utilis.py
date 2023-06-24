import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Set, Callable, Union, Optional
from abc import ABC, abstractmethod
from time import time as stamp

class differential:
    def __init__(self, iterdim: int=0, order: int=1, batch: bool=False):
        """
        This class can differentiate tensors into tensors.
        1. numpy array dimension
            (1) scalar : 0
            (2) vector : 1
            (3) matrix : 2
            (4) tensor : 3
            ...
    
        2. order
            (1) first order : 1
            (2) second order : 2
            ...
    
        3. batch
            (1) True
            (2) False
        
        Input tensor is instance of numpy array, also output tensor is instance of numpy array.
        Functions have to use numpy array.
        
        There are examples of functions below.
        
        <function examples>
        def function1(x: np.array) -> np.array: #input : 0, output : 0
            return x[0]**2

        def function2(x: np.array) -> np.array: #input : 1, output : 0
            return x[0]**2 + 2*x[1]

        def function3(x: np.array) -> np.array: #input : 0, output : 1
            return np.array([3*x[0], x[0]**2])

        def function4(x: np.array) -> np.array: #input : 1, output : 1
            return np.array([x[0]**2 + 2*x[1], 2*x[0] - 3*x[1], x[0]**3 - 3*(x[1]**2)])

        def function5(x: np.array) -> np.array: #input : 1, output : 2
            return np.array([[x[0]**2 - 3*x[1], x[0]**3 - 3*(x[1]**2)],
                             [x[0]**2, x[1]**3]])
                             
        def function6(x: np.array) -> np.array: #input : 2, output : 0
            return np.array(x[0,0]**2 + x[0,1] + 2*x[1,0] - 3*x[1,1])
        
        function1 is univariable scalar function.
        function2 is multivariable scalar function.
        function3 is univariable vector function.
        function4 is multivariable vector function.
        ...
        
        Shape of differential is input + output.
        For example, function4 has a shape of 2 as result. Therefore result is matrix.
        
        input_parameter_dimension is dimension of input parameter.
        For example, there is x = [1, 2, 3]. If you want to differentiate into each element, input_parameter_dimension = 0(scalar).
        If you want to differentiate the entire x array into one element, input_parameter_dimension = 1(vector).
        
        """
        self.iterdim = iterdim
        self.order = order
        self.batch = batch
        self.eps = 1e-5
        
    def __call__(self, f: Callable, x: np.array, args: Optional[np.array]=None) -> np.array:
        self.x = x.astype(np.float64)
        self.y = np.array(f(x, args)).astype(np.float64)
        
        batch_grad = []

        if self.batch:
            for idx_batch, data in enumerate(x):
                constant = args[idx_batch] if args is not None else None
                batch_grad.append(self.gradient(f, data, constant))
            return np.array(batch_grad)
        
        else:
            return self.gradient(f, x, args)
    
    def gradient(self, f: Callable, x: np.array, args: Optional[np.array]=None) -> np.array:
        x = x.astype(np.float64)
        eps = 1e-4
        itt = differential.slicing_tensor(x, self.iterdim)

        tensorX = []
        constant = []
        grad = []
        for idx, value in itt:
            tensorX.append(value)
            constant.append(args[idx] if args is not None else None)

        constant = np.array(constant)
        tensorX = np.array(tensorX)
        
        for idx, value in enumerate(tensorX):
            temp = tensorX[idx].copy()

            #plus eps
            tensorX[idx] = temp + eps
            f1 = f(tensorX, constant)
            
            #minus eps
            tensorX[idx] = temp - eps
            f2 = f(tensorX, constant)
            #diff
            result = (f1 - f2) / (2 * eps)

            grad.append(result)
            #restoration
            tensorX[idx] = temp

        grad = np.array(grad)
        
        try:
            grad = np.squeeze(grad)
        except:
            pass
        return grad
    
    @staticmethod
    def slicing_tensor(x: np.array, dim: int) -> Tuple[Tuple, np.float64]:
        dim = len(x.shape) - dim
        if dim < 0:
            raise Exception("The extracted tensor cannot be larger in dimension than the original tensor.")

        input_shape = x.shape[0:dim]

        x_shape = np.zeros(input_shape)
        x_dim = np.zeros(dim)
        it = np.nditer(x_shape, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index
            it.iternext()
            yield idx, x[idx]    
    
def Initialization(layer_type: str, activation_type: str, weight_shape: Tuple, init_type: str="optim") -> Tuple[np.array, np.array]:
    """Initializing weight and bias"""
    relu_form = ['linear','relu','leakey_relu','elu','prelu']
    curve_form = ['sigmoid','tanh','softmax']
    init_type_list = set(["random", "Xavier", "He", "optim"])
    
    if init_type not in init_type_list:
        raise Exception("Can't initializing. You should specify the init type.")
    
    #random_uniform
    if init_type == "random":
        if layer_type[1] == 'dense':
            W = np.random.rand(*weight_shape)
            b = np.zeros(weight_shape[1])

        elif layer_type[1] == 'conv2d':
            W = np.random.rand(*weight_shape)
            b = np.zeros((1, weight_shape[0]))
        return W, b
    
    #Xavier initialize for dense layer
    if layer_type[1] == 'dense' and activation_type in curve_form:
        W = np.random.randn(*weight_shape) * np.sqrt(1.0 / weight_shape[1])
        b = np.zeros(weight_shape[1])
    
    #He initialize for dense layer
    elif layer_type[1] == 'dense' and activation_type in relu_form:
        W = np.random.randn(*weight_shape) * np.sqrt(2.0 / weight_shape[1])
        b = np.zeros(weight_shape[1])
    
    #Xavier initialize for convolutional layer
    elif layer_type[1] == 'conv2d' and activation_type in curve_form:
        init_in = weight_shape[1]*weight_shape[2]*weight_shape[3]
        W = np.random.randn(*weight_shape) * np.sqrt(1.0 / init_in)
        b = np.zeros((1, weight_shape[0]))
        
    #He initialize for convolutional layer
    elif layer_type[1] == 'conv2d' and activation_type in relu_form:
        init_in = weight_shape[1]*weight_shape[2]*weight_shape[3]
        W = np.random.randn(*weight_shape) * np.sqrt(2.0 / init_in)
        b = np.zeros((1, weight_shape[0]))
    
    else:
        raise Exception("Can't initializing")
    return W, b

def im2col(image: Tuple, flt_h: int, flt_w: int, padding: int=0, stride: int=1) -> np.array:
    """Image to column convertor"""
    num, ch, img_h, img_w = image.shape
    padimg = np.pad(image, [(0,0),(0,0),(padding,padding),(padding,padding)], "constant")
    
    out_h = (img_h - flt_h + 2*padding) // stride + 1
    out_w = (img_w - flt_w + 2*padding) // stride + 1
    
    col_mat = np.zeros((num, ch, flt_h, flt_w, out_h, out_w))
    
    for h in range(flt_h):
        h_end = h + stride*out_h
        for w in range(flt_w):
            w_end = w + stride*out_w
            col_mat[:, :, h, w, :, :] = padimg[:, :, h:h_end:stride, w:w_end:stride]
        
    col_mat = col_mat.transpose(1,2,3,0,4,5).reshape(ch*flt_h*flt_w,num*out_h*out_w)
    return col_mat

def col2im(col_mat: np.array, image_shape: Tuple, flt_h: int, flt_w: int, padding: int=0, stride:int=1) -> np.array:
    """Column to image convertor"""
    num, ch, img_h, img_w = image_shape
    
    out_h = (img_h - flt_h + 2*padding) // stride + 1
    out_w = (img_w - flt_w + 2*padding) // stride + 1 
    
    col_mat = col_mat.reshape(ch, flt_h, flt_w, num, out_h, out_w).transpose(3,0,1,2,4,5)
    image = np.zeros((num, ch, img_h + 2*padding + stride-1, img_w + 2*padding + stride-1))
    
    for h in range(flt_h):
        h_end = h + stride*out_h
        for w in range(flt_w):
            w_end = w + stride*out_w
            image[:,:,h:h_end:stride,w:w_end:stride] += col_mat[:,:,h,w,:,:]
    return image[:,:,padding:img_h + padding, padding:img_w + padding]

def OneHot(label_data: np.array, num_class: int) -> np.array:
    """OneHot encoding function"""
    return np.identity(num_class)[label_data]

#loss function
def Loss_Cross_Entropy(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    eps = 1e-4
    return -np.sum(true_y * np.log(pred_y + eps)) / batch_size

def Loss_Cross_Entropy_bin(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    eps = 1e-4
    return (-true_y * np.log(pred_y + eps) - (1 - true_y) * np.log(1 - pred_y + eps)) / batch_size

def Loss_MSE(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    if pred_y.ndim == 1:
        batch_size = 1
    return (true_y - pred_y)**2 / batch_size

def Loss_RMSE(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    if pred_y.ndim == 1:
        batch_size = 1
    return np.sqrt(np.sum(Loss_MSE(pred_y, true_y))) / batch_size
    
#built-in derivative
def Loss_Cross_Entropy_diff(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    return - (true_y / pred_y) / batch_size

def Loss_Cross_Entropy_bin_diff(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    return -((true_y / pred_y) - ((1 - true_y) / (1 - pred_y))) / batch_size

def Loss_MSE_diff(pred_y: np.array, true_y: np.array) -> np.array:
    batch_size = len(pred_y)
    return (pred_y - true_y) / batch_size

def Loss_Softmax_with_CrossEntropy_diff(pred_y: np.array, true_y: np.array) -> np.array:
    return pred_y - true_y

def process_visualizing(current: int, total: int) -> int:
    width = 40
    percent = round(current / total * 100)
    visual_percent = round(width * (current / total))
    print(f"\rmini batch : {percent}%","[" + "="*visual_percent + ">" + " "*(width-visual_percent) + "]" ,end="")
    return width



#If using python eval function, there is possibility of hacking. So this library uses dict.
activation_dict = {'linear' : "Linear()",
                   'sigmoid' : "Sigmoid()",
                   'tanh' : "tanh()",
                   'softmax' : "Softmax()",
                   'relu' : "ReLU()",
                   'leakey_relu' : "Leaky_ReLU()",
                   'elu' : "ELU()",
                   'prelu' : "PReLU()"}

optimizer_dict = {'SGD' : "SGD()",
                  'Momentum' : "Momentum()",
                  'Adagrad' : "AdaGrad()",
                  'RMSprop' : "RMSprop()",
                  'Adam' : "Adam()"}

Loss_dict = {"MSE" : "Loss_MSE",
            "CrossEntropy" : "Loss_Cross_Entropy", 
            "CrossEntropy_binary" : "Loss_Cross_Entropy_bin", 
            "Softmax_with_CrossEntropy" : "Loss_Softmax_with_CrossEntropy"}

metrics_set = {"rmse", "accuracy"}