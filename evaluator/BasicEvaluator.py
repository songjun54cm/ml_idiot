__author__ = 'SongJun-Dell'
import numpy as np

class BasicEvaluator(object):
    def __init__(self):
        pass

    def form_logical_value(self, gth, pred, threshold=0.5):
        if gth.dtype != 'bool':
            logical_gth = gth>threshold
            logical_pred = pred>threshold
        else:
            logical_gth = gth
            logical_pred = pred
        return logical_gth, logical_pred
