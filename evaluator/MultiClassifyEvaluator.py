__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/8
import argparse
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator
from ml_idiot.evaluator import EvaluatorMetrics


class MultiClassifyEvaluator(BasicEvaluator):
    def __init__(self, config=None):
        super(MultiClassifyEvaluator, self).__init__(config)
        self.metric_func_mapping = {
            "accuracy": EvaluatorMetrics.classify_accuracy,
        }
        self.top_metric = config.get("top_metric")

