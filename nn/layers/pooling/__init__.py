__author__ = 'SongJun-Dell'
import numpy as np


def max_pooling(x, axis=0):
    return np.max(x, axis=axis, keepdims=True)

def average_pooling(x, axis=0):
    return np.mean(x, axis=axis, keepdims=True)
