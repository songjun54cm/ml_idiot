__author__ = 'SongJun-Dell'
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator
from collections import OrderedDict
class BasicTester(object):
    def __init__(self):
        self.evaluator = BasicEvaluator()

    def test_on_split(self, model, data_provider, split):
        split_data = data_provider.splits[split]
        res = self.test_batch(model, split_data)
        return res

    def get_metrics(self, res, metrics):
        metric_res = OrderedDict()
        if type(metrics) == list:
            for met in metrics:
                met_str, met_value = self.get_metric_value(res, met)
                metric_res[met_str] = met_value
                # print met_value
        else:
            met_str, met_value = self.get_metric_value(res, metrics)
            metric_res[met_str] = met_value
        return metric_res

    # need to be implemented
    def init_tester(self):
        raise NotImplementedError('init tester not implemented')

    def test_batch(self, model, data_batch):
        raise NotImplementedError('test batch not implemented.')

    def get_metric_value(self, res, metrics):
        raise NotImplementedError('get_metrics not implemented.')