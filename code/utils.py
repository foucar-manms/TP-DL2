import sys
import os
import time
import numpy as np


CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'

IMAGE_SHAPE = {
    'binary_alpha_digits':(20,16),
    'MNIST':(28,28)
}

def char2index(char):
    assert len(char) == 1 and char in CHARS, (
        "char should be either a letter in lower case or a digit from 0 to 9, in either "
        "case, it should be a string. {} is invalid".format(char)
    )
    return CHARS.find(char)

def index2char(index):
    assert 0 <= index <= 35, (
        "index should be an integer between 0 and 35, not {}".format(index)
    )
    return CHARS[index]

def vec2img(flat_img, matrix_shape='binary_alpha_digits'):
    """ turn a flattened image back to its original shape (specified by matrix_shape) """
    if isinstance(matrix_shape, type('')):
        try:
            matrix_shape = IMAGE_SHAPE[matrix_shape]
        except KeyError:
            print("unknown matrix format : {}".format(matrix_shape))
            sys.exit()
    return flat_img.reshape(matrix_shape)

def sigmoid(x):
    """ return the sigmoid value evaluated at x """
    return 1 / (1 + np.exp(-x))

def cross_entropy(x, y):
    """ return the cross entropy value between the distributions x and y """
    if len(x.shape) == 1: x = np.expand_dims(x, axis=0)
    if len(y.shape) == 1: x = np.expand_dims(y, axis=0)
    return - np.sum(y * np.log(x), axis=1)

def monitor_experience(out_file):
    """ write configurations and results of a DNN in a .txt file """
    pass