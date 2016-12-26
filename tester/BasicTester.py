__author__ = 'SongJun-Dell'
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator

class BasicTester(object):
    def __init__(self):
        self.evaluator = BasicEvaluator()

    def test_on_split(self, model, data_provider, split, metrics):
        split_data = data_provider.splits[split]
        res = self.test_batch(model, split_data, metrics)
        return res

    # need to be implemented
    def test_batch(self, model, data_batch, metrics):
        raise NotImplementedError('test batch not implemented.')


    def detect_to_save(self, res, model):
        raise NotImplementedError('detect_to_save not implemented.')

    def get_metrics(self, res, metrics):
        raise NotImplementedError('get_metrics not implemented.')