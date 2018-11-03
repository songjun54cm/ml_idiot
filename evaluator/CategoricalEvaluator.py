__author__ = 'JunSong<songjun54cm@gmail.com>'
from BasicEvaluator import BasicEvaluator
import numpy as np


class CategoricalEvaluator(BasicEvaluator):
    def __init__(self):
        super(CategoricalEvaluator, self).__init__()

    def categorical_accuracy(self, gth, pred):
        return np.mean(np.equal(np.argmax(gth, axis=-1),
                              np.argmax(pred, axis=-1)))