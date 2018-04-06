__author__ = 'JunSong'
import numpy as np
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator

class MultiLabelEvaluator(BasicEvaluator):
    def __init__(self):
        super(MultiLabelEvaluator, self).__init__()
        self.metrics = ['Accuracy',
                        'F1-Score',
                        'TruePositiveRate',
                        'TrueNegativeRate',
                        'FalsePositiveRate',
                        'FalseNegativeRate']

    def evaluate(self, gth_vals, pred_vals, metrics=None):
        res = {}
        if isinstance(gth_vals, np.ndarray):
            logical_gth_vals, logical_pred_vals = self.form_logical_value(gth_vals, pred_vals)
            res['Precision'] = self.logical_precision(logical_gth_vals, logical_pred_vals)
            res['Recall'] = self.logical_recall(logical_gth_vals, logical_pred_vals)
            res['F1-Score'] = self.logical_f1_score(p=res['Precision'], r=res['Recall'])

        elif isinstance(gth_vals, list):
            res['Precision'] = self.precision(gth_vals, pred_vals)
            res['Recall'] = self.recall(gth_vals, pred_vals)
            res['F1-Score'] = self.f1_score(p=res['Precision'], r=res['Recall'])
        else:
            raise BaseException('ground-truth values type error.')
        return res

    def precision(self, gth_vals, pred_vals):
        if isinstance(gth_vals[0], int):
            gth_set = set(gth_vals)
            pred_set = set(pred_vals)
            pred_true_num = len(pred_set)
            if pred_true_num == 0:
                return 0.0
            else:
                tt_num = len(pred_set.intersection(gth_set))
        elif isinstance(gth_vals[0], list):
            pred_true_num = 0
            tt_num = 0
            for gvs, pvs in zip(gth_vals, pred_vals):
                pred_set = set(pvs)
                pred_true_num += len(pred_set)
                tt_num += len(pred_set.intersection(set(gvs)))
        else:
            raise TypeError('ground truth values type error.')
        precision_val = float(tt_num) / float(pred_true_num)
        return precision_val

    def logical_precision(self, logical_gth, logical_pred):
        pred_true_num = np.where(logical_pred)[0].shape[0]
        if pred_true_num == 0:
            return 0.0
        else:
            tt_num = np.where(np.logical_and(logical_gth, logical_pred))[0].shape[0]
            precision = float(tt_num) / float(pred_true_num)
        return precision

    def recall(self, gth_vals, pred_vals):
        if isinstance(gth_vals[0], int):
            gset = set(gth_vals)
            gth_true_num = len(gset)
            tt_num = gset.intersection(set(pred_vals))
        elif isinstance(gth_vals[0], list):
            gth_true_num = 0
            tt_num = 0
            for gvs, pvs in zip(gth_vals, pred_vals):
                gset = set(gvs)
                gth_true_num += len(gset)
                tt_num += len(gset.intersection(set(pvs)))
        else:
            raise TypeError('ground truth values type error.')
        recall_val = float(tt_num) / float(gth_true_num)
        return recall_val

    def logical_recall(self, gth, pred):
        gth_true_num = np.where(gth)[0].shape[0]
        tt_num = np.where(np.logical_and(gth, pred))[0].shape[0]
        recall_val = float(tt_num) / float(gth_true_num)
        return recall_val

    def f1_score(self, p=None, r=None, gth_vals=None, pred_vals=None):
        if p is None:
            p = self.precision(gth_vals, pred_vals)
            r = self.recall(gth_vals, pred_vals)
        p_plus_r = p + r
        if p_plus_r == 0:
            return 0.0
        else:
            f1 = 2*p*r/(p+r)
        return f1

    def logical_f1_score(self, p=None, r=None, gth=None, pred=None):
        if p is None:
            p = self.logical_precision(gth, pred)
            r = self.logical_recall(gth, pred)
        p_plus_r = p + r
        if p_plus_r == 0:
            return 0.0
        else:
            f1 = 2*p*r/(p+r)
        return f1

    def accuracy(self, gth_vals, pred_vals):
        if isinstance(gth_vals[0], int):
            gth_set = set(gth_vals)
            pred_set = set(pred_vals)
            pred_true_num = len(pred_set)
            if pred_true_num == 0:
                return 0.0
            else:
                tt_num = len(pred_set.intersection(gth_set))
                accuracy = float(tt_num) / float(pred_true_num)
        elif isinstance(gth_vals[0], list):
            pred_true_num = 0
            tt_num = 0
            for gvs, pvs in zip(gth_vals, pred_vals):
                pred_set = set(pvs)
                pred_true_num += len(pred_set)
                tt_num += len(pred_set.intersection(set(gvs)))
            accuracy = float(tt_num) / float(pred_true_num)
        else:
            raise TypeError('ground truth values type error.')

        return accuracy

    def logical_accuracy(self, gth, pred):
        pred_true_num = np.where(pred)[0].shape[0]
        if pred_true_num == 0:
            return 0.0
        else:
            tt_num = np.where(np.logical_and(gth, pred))[0].shape[0]
            accuracy = float(tt_num) / float(pred_true_num)
        return accuracy

    def true_positive_rate(self, gth_vals, pred_vals):
        if isinstance(gth_vals[0], int):
            gset = set(gth_vals)
            gth_true_num = len(gset)
            tt_num = gset.intersection(set(pred_vals))
        elif isinstance(gth_vals[0], list):
            gth_true_num = 0
            tt_num = 0
            for gvs, pvs in zip(gth_vals, pred_vals):
                gset = set(gvs)
                gth_true_num += len(gset)
                tt_num += len(gset.intersection(set(pred_vals)))
        else:
            raise TypeError('ground truth values type error.')
        ttrate = float(tt_num) / float(gth_true_num)
        return ttrate

    def logical_true_positive_rate(self, gth, pred):
        gth_true_num = np.where(gth)[0].shape[0]
        tt_num = np.where(np.logical_and(gth, pred))[0].shape[0]
        ttrate = float(tt_num) / float(gth_true_num)
        return ttrate

    def true_negative_rate(self, gth_vals, pred_vals):
        raise NotImplementedError

    def logical_true_negative_rate(self, gth, pred):
        not_logical_gth = np.logical_not(gth)
        gth_false_num = np.where(not_logical_gth)[0].shape[0]
        tn_num = np.where(np.logical_and(not_logical_gth, ~pred))[0].shape[0]
        tn_rate = float(tn_num) / float(gth_false_num)
        return tn_rate

    def false_positive_rate(self, gth_vals, pred_vals):
        return 1 - self.true_positive_rate(gth_vals, pred_vals)

    def logical_false_positive_rate(self, logical_gth, logical_pred):
        gth_true_num = np.where(logical_gth)[0].shape[0]
        tf_num = np.where(np.logical_and(logical_gth, ~logical_pred))[0].shape[0]
        tf_rate = float(tf_num) / float(gth_true_num)
        return tf_rate

    def false_negative_rate(self, gth_vals, pred_vals):
        raise NotImplementedError

    def logical_false_negative_rate(self, logical_gth, logical_pred):
        not_logical_gth = np.logical_not(logical_gth)
        gth_false_num = np.where(not_logical_gth)[0].shape[0]
        ft_num = np.where(np.logical_and(not_logical_gth, logical_pred))[0].shape[0]
        ft_rate = float(ft_num) / float(gth_false_num)
        return ft_rate

