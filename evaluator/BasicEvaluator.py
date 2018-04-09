__author__ = 'SongJun-Dell'
import numpy as np
import abc


class BasicEvaluator(object):
    def __init__(self, metrics=None):
        self.metric_func_mapping = {}
        self.metrics = None

    def init_metric(self, metrics):
        if metrics is None:
            self.metrics = list(self.metric_func_mapping.keys())
        else:
            self.metrics = metrics

    def evaluate_prepare(self):
        pass

    def evaluate(self, gth_vals, pred_vals, metrics=None):
        """
        evaluate predict results.
        :param gth_vals:    ground-truth values.
        :param pred_vals:   predicted values
        :return:    metrics = {metric_name: metric_value}
        """
        self.evaluate_prepare()
        if metrics is None:
            metrics = self.metrics
        metric_res = {}
        for met_name in metrics:
            metric_res[met_name] = self.metric_func_mapping[met_name](gth_vals, pred_vals)
        return metric_res


    def form_logical_value(self, gth, pred, threshold=0.5):
        if gth.dtype != 'bool':
            logical_gth = gth>threshold
            logical_pred = pred>threshold
        else:
            logical_gth = gth
            logical_pred = pred
        return logical_gth, logical_pred
