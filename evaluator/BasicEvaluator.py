__author__ = 'SongJun-Dell'
import numpy as np
import abc

class BasicEvaluator(object):
    metrics = []
    def __init__(self, metrics=None):
        pass

    @abc.abstractmethod
    def evaluate(self, gth_vals, pred_vals):
        """
        evaluate predict results.
        :param gth_vals:    ground-truth values.
        :param pred_vals:   predicted values
        :return:    metrics = {metric_name: metric_value}
        """

    def form_logical_value(self, gth, pred, threshold=0.5):
        if gth.dtype != 'bool':
            logical_gth = gth>threshold
            logical_pred = pred>threshold
        else:
            logical_gth = gth
            logical_pred = pred
        return logical_gth, logical_pred
