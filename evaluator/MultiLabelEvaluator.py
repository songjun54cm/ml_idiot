__author__ = 'JunSong'
import numpy as np
from BasicEvaluator import BasicEvaluator

class MultiLabelEvaluator(BasicEvaluator):
    def __init__(self):
        super(MultiLabelEvaluator, self).__init__()

    def form_logical_value(self, gth, pred):
        if gth.dtype != 'bool':
            logical_gth = gth>0.5
            logical_pred = pred>0.5
        else:
            logical_gth = gth
            logical_pred = pred
        return logical_gth, logical_pred

    def f1_score(self, gth, pred):
        p = self.accuracy(gth, pred)
        r = self.true_positive_rate(gth, pred)
        p_plus_r = p + r
        if p_plus_r == 0:
            return 0.0
        else:
            f1 = 2*p*r/(p+r)
        return f1

    def true_positive_rate(self, gth, pred):
        logical_gth, logical_pred = self.form_logical_value(gth, pred)
        gth_true_num = np.where(logical_gth)[0].shape[0]
        tt_num = np.where(np.logical_and(logical_gth, logical_pred))[0].shape[0]
        ttrate = float(tt_num) / float(gth_true_num)
        return ttrate

    def true_negative_rate(self, gth, pred):
        logical_gth, logical_pred = self.form_logical_value(gth, pred)
        not_logical_gth = np.logical_not(logical_gth)
        gth_false_num = np.where(not_logical_gth)[0].shape[0]
        tn_num = np.where(np.logical_and(not_logical_gth, ~logical_pred))[0].shape[0]
        tn_rate = float(tn_num) / float(gth_false_num)
        return tn_rate

    def false_positive_rate(self, gth, pred):
        logical_gth, logical_pred = self.form_logical_value(gth, pred)
        gth_true_num = np.where(logical_gth)[0].shape[0]
        tf_num = np.where(np.logical_and(logical_gth, ~logical_pred))[0].shape[0]
        tf_rate = float(tf_num) / float(gth_true_num)
        return tf_rate

    def false_negative_rate(self, gth, pred):
        logical_gth, logical_pred = self.form_logical_value(gth, pred)
        not_logical_gth = np.logical_not(logical_gth)
        gth_false_num = np.where(not_logical_gth)[0].shape[0]
        ft_num = np.where(np.logical_and(not_logical_gth, logical_pred))[0].shape[0]
        ft_rate = float(ft_num) / float(gth_false_num)
        return ft_rate

    def accuracy(self, gth, pred):
        logical_gth, logical_pred = self.form_logical_value(gth, pred)
        pred_true_num = np.where(logical_pred)[0].shape[0]
        if pred_true_num == 0:
            return 0.0
        else:
            tt_num = np.where(np.logical_and(logical_gth, logical_pred))[0].shape[0]
            accuracy = float(tt_num) / float(pred_true_num)
        return accuracy

