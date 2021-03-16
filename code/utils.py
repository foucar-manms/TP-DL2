import sys
import numpy as np
from numpy.matrixlib.defmatrix import matrix


CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'

IMAGE_SHAPE = {
    'binary_alpha_digits':(20,16),
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
    """ return the sigmoid value function evaluated at x """
    return 1 / (1 + np.exp(-x))
