__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/8
import argparse
from ml_idiot.evaluator.BinaryClassifyEvaluator import BinaryClassifyEvaluator


class LgbmGbdtEvaluator(BinaryClassifyEvaluator):
    def __init__(self, config):
        super(LgbmGbdtEvaluator, self).__init__(config)
