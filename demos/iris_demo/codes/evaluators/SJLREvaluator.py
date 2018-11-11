__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.evaluator.BinaryClassifyEvaluator import BinaryClassifyEvaluator


class SJLREvaluator(BinaryClassifyEvaluator):
    def __init__(self, config):
        super(SJLREvaluator, self).__init__(config)
