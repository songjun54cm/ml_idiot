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

    def detect_to_save(self, res, model):
        metric_score = res['metrics'].get(self.top_metric_name)
        if metric_score > self.top_metric:
            self.top_metric = metric_score
            save_tag = True
        else:
            save_tag = False
        # cp_sufix = 'accuracy_%.3f_tt_%.3f_.pkl' % (accuracy, truetrue)
        cp_sufix = '%s_%.6f_.pkl' % (self.top_metric_name, metric_score)
        return save_tag, cp_sufix

    # need to be implemented
    def init_tester(self):
        self.top_metric = 0
        self.top_metric_name = ''
        raise NotImplementedError('init tester not implemented')

    def test_batch(self, model, data_batch):
        raise NotImplementedError('test batch not implemented.')

    def get_metric_value(self, res, metrics):
        raise NotImplementedError('get_metrics not implemented.')