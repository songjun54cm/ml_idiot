__author__ = 'JunSong<songjun54cm@gmail.com>'
from ml_idiot.tester.BasicTester import BasicTester
from ml_idiot.evaluator.CategoricalEvaluator import CategoricalEvaluator


class CategoricalTester(BasicTester):
    def __init__(self):
        super(CategoricalTester, self).__init__()
        self.evaluator = CategoricalEvaluator()
        self.init_tester()

    def init_tester(self):
        self.top_metric_name = 'categorical_accuracy'
        self.top_metric = -1.0

    def get_metric_value(self, res, met):
        gth_state_feas = res['gth_feas']
        pred_state_feas = res['pred_feas']
        if met in ['categorical_accuracy']:
            return met, self.evaluator.categorical_accuracy(gth_state_feas, pred_state_feas)
        else:
            logging.info('metric: %s' % met)
            raise KeyError('metric name error!')