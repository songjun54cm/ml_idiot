__author__ = 'SongJun-Dell'
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator

class BasicTester(object):
    def __init__(self):
        self.evaluator = BasicEvaluator()

    # need to be implemented
    def test_on_split(self, model, data_provider, split):
        raise NotImplementedError

    def detect_to_save(self, res, model):
        raise NotImplementedError

    def get_metrics(self, res, metrics):
        raise NotImplementedError