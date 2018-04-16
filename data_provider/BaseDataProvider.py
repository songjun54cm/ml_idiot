__author__ = 'JunSong<songjun54cm@gmail.com>'
import logging
import numpy as np


class BaseDataProvider(object):
    def __init__(self):
        pass

    def summary(self):
        for key, val in self.__dict__.items():
            logging.info("%s, %s, size: %s" % (self.__class__.__name__, key, self.get_size(val)))

    @staticmethod
    def get_size(val):
        if isinstance(val, np.ndarray):
            return str(val.shape)
        if type(val) in [dict, list, set, tuple, str]:
            return str(len(val))
        if type(val) in [int, float, bool]:
            return str(val)