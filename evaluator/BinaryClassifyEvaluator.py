__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator
from ml_idiot.evaluator import EvaluatorMetrics


class BinaryClassifyEvaluator(BasicEvaluator):
    def __init__(self, metrics=None):
        super(BinaryClassifyEvaluator, self).__init__()
        self.metric_func_mapping = {
            "accuracy": EvaluatorMetrics.classify_accuracy,
            "auc": EvaluatorMetrics.auc
        }
