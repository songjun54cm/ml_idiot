__author__ = 'JunSong'
import numpy as np
from BasicEvaluator import BasicEvaluator

class MultiLabelEvaluator(BasicEvaluator):
    def __init__(self):
        super(MultiLabelEvaluator, self).__init__()

    def true_true_rate(self, gth, pred):
        if gth.dtype != 'bool':
            logical_gth = gth>0.5
            logical_pred = pred>0.5
        else:
            logical_gth = gth
            logical_pred = pred
        gth_true_num = np.where(logical_gth)[0].shape[0]
        tt_num = np.where(np.logical_and(logical_gth, logical_pred))[0].shape[0]
        ttrate = float(tt_num) / float(gth_true_num)
        return ttrate

    def true_false_rate(self, gth, pred):
        if gth.dtype != 'bool':
            logical_gth = gth>0.5
            logical_pred = pred>0.5
        else:
            logical_gth = gth
            logical_pred = pred
        gth_true_num = np.where(logical_gth)[0].shape[0]
        tf_num = np.where(np.logical_and(logical_gth, ~logical_pred))[0].shape[0]
        tf_rate = float(tf_num) / float(gth_true_num)
        return tf_rate

    def false_true_rate(self, gth, pred):
        if gth.dtype != 'bool':
            logical_gth = gth>0.5
            logical_pred = pred>0.5
        else:
            logical_gth = gth
            logical_pred = pred
        not_logical_gth = np.logical_not(logical_gth)
        gth_false_num = np.where(not_logical_gth)[0].shape[0]
        ft_num = np.where(np.logical_and(not_logical_gth, logical_pred))[0].shape[0]
        ft_rate = float(ft_num) / float(gth_false_num)
        return ft_rate

