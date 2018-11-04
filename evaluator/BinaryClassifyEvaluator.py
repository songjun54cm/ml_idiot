__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator
from ml_idiot.evaluator import EvaluatorMetrics


class BinaryClassifyEvaluator(BasicEvaluator):
    def __init__(self, config=None):
        super(BinaryClassifyEvaluator, self).__init__(config)
        self.metric_func_mapping = {
            "accuracy": EvaluatorMetrics.classify_accuracy,
            "auc": EvaluatorMetrics.auc
        }
        self.top_metric = config.get("top_metric", "auc")
