__author__ = 'SongJun-Dell'
from collections import OrderedDict
from ml_idiot.evaluator.MultiLabelEvaluator import MultiLabelEvaluator
import numpy as np

class HMMTester(object):
    def __init__(self):
        self.evaluator = MultiLabelEvaluator()

    def test(self, model, test_data, id_to_fea, metrics):
        obs_seqs = test_data['obs_seqs']
        state_seqs = test_data['state_seqs']
        gth_states = list()
        pred_states = list()

        for si in xrange(len(obs_seqs)):
            obss = obs_seqs[si]
            states = state_seqs[si]
            pred_ss = model.predict_states(obss, states)
            gth_states += states
            pred_states = np.concatenate([pred_states, pred_ss])

        res_feas = {
            'gth_feas': id_to_fea[gth_states],
            'pred_feas': id_to_fea[pred_states.astype(int)]
        }

        res = self.get_metrics(res_feas, metrics)
        return res

    def get_metrics(self, res, metrics):
        gth_state_feas = res['gth_feas']
        pred_state_feas = res['pred_feas']
        res = OrderedDict()
        if type(metrics) == list:
            for met in metrics:
                met_str, met_value = self.get_metric_value(met, gth_state_feas, pred_state_feas)
                res[met_str] = met_value
                # print met_value
        else:
            met_str, met_value = self.get_metric_value(metrics, gth_state_feas, pred_state_feas)
            res[met_str] = met_value
        return res

    def get_metric_value(self, met, gth_state_feas, pred_state_feas):
        if met == 'f1':
            return 'F1', self.evaluator.f1_score(gth_state_feas, pred_state_feas)
        elif met == 'tp':
            return 'True-Positive', self.evaluator.true_positive_rate(gth_state_feas, pred_state_feas)
        elif met == 'tn':
            return 'True-Negative', self.evaluator.true_negative_rate(gth_state_feas, pred_state_feas)
        elif met == 'fp':
            return 'False-Positive', self.evaluator.false_positive_rate(gth_state_feas, pred_state_feas)
        elif met == 'ft':
            return 'False-Negative', self.evaluator.false_negative_rate(gth_state_feas, pred_state_feas)
        elif met == 'accuracy':
            return 'Accuracy', self.evaluator.accuracy(gth_state_feas, pred_state_feas)
        else:
            raise StandardError('metric name error!')