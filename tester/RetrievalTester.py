"""
Author: songjun
Date: 2018/4/9
"""
import logging
from ml_idiot.tester.BasicTester import BasicTester
from ml_idiot.evaluator.RetrievalEvaluator import RetrievalEvaluator


class RetrievalTester(BasicTester):
    def __init__(self, config):
        super(RetrievalTester, self).__init__(config)
        self.evaluator = RetrievalEvaluator()
