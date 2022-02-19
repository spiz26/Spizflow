import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time as stamp
from abc import ABC, abstractmethod

def Initialization(layer_type, activation_type, weight_shape):
    """Initializing weight and bias"""
    relu_form = ['linear','relu','leakey_relu','elu','prelu']
    curve_form = ['sigmoid','tanh','softmax']
    
    if layer_type[1] == 'dense' and activation_type in curve_form:
        W = np.random.randn(*weight_shape) * np.sqrt(1.0 / weight_shape[1])
        b = np.random.randn(weight_shape[1])
        
    elif layer_type[1] == 'dense' and activation_type in relu_form:
        W = np.random.randn(*weight_shape) * np.sqrt(2.0 / weight_shape[1])
        b = np.random.randn(weight_shape[1])
        
    elif layer_type[1] == 'conv2d' and activation_type in curve_form:
        init_in = weight_shape[1]*weight_shape[2]*weight_shape[3]
        W = np.random.randn(*weight_shape) * np.sqrt(1.0 / init_in)
        b = np.random.randn(1, weight_shape[0]) * 0.1

    elif layer_type[1] == 'conv2d' and activation_type in relu_form:
        init_in = weight_shape[1]*weight_shape[2]*weight_shape[3]
        W = np.random.randn(*weight_shape) * np.sqrt(2.0 / init_in)
        b = np.random.randn(1, weight_shape[0]) * 0.1
    
    else:
        raise Exception("Can't initializing")
    return W, b

def im2col(image, flt_h, flt_w, padding=0, stride=1):
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

def col2im(col_mat, image_shape, flt_h, flt_w, padding=0, stride=1):
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

def OneHot(label_data, num_class):
    """OneHot encoding function"""
    return np.identity(num_class)[label_data]

def Loss_Cross_Entropy(pred_y, true_y, batch_size):
    return -np.sum(true_y * np.log(pred_y + 1e-8)) / batch_size

def Loss_RMSE(pred_y, true_y):
    return np.sqrt((pred_y - true_y)**2) / 2.0