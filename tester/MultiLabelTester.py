__author__ = 'SongJun-Dell'
from collections import OrderedDict
from ml_idiot.tester.BasicTester import BasicTester
from ml_idiot.evaluator.MultiLabelEvaluator import MultiLabelEvaluator
import numpy as np

class MultiLabelTester(BasicTester):
    def __init__(self):
        super(MultiLabelTester, self).__init__()
        self.evaluator = MultiLabelEvaluator()
        self.init_tester()

    def init_tester(self):
        self.top_f1 = 0

    def test_batch(self, model, batch_data):
        gth_feas = [data['target_feas'] for data in batch_data]
        preds = model.predict_batch(batch_data)
        losses = list()
        for gfea, pfea in zip(gth_feas, preds):
            losses.append(model.loss_one_sample(gfea, pfea))
        loss = np.mean(losses)

        res = {
            'loss': loss,
            'sample_num': len(batch_data),
            'gth_feas': np.concatenate(gth_feas),
            'pred_feas': np.concatenate(preds) if type(preds) is list else preds
        }
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
        if met in ['f1', 'F1']:
            return met, self.evaluator.f1_score(gth_state_feas, pred_state_feas)
        elif met in ['tp', 'TruePositive']:
            return met, self.evaluator.true_positive_rate(gth_state_feas, pred_state_feas)
        elif met in ['tn', 'TrueNegative']:
            return met, self.evaluator.true_negative_rate(gth_state_feas, pred_state_feas)
        elif met in ['fp', 'FalsePositive']:
            return met, self.evaluator.false_positive_rate(gth_state_feas, pred_state_feas)
        elif met in ['fn', 'FalseNegative']:
            return met, self.evaluator.false_negative_rate(gth_state_feas, pred_state_feas)
        elif met in ['acc', 'accuracy', 'Accuracy']:
            return met, self.evaluator.accuracy(gth_state_feas, pred_state_feas)
        else:
            print 'metric: %s' % met
            raise StandardError('metric name error!')

    def detect_to_save(self, res, model):
        # accuracy = res['Accuracy']
        # truetrue = res['True-True']
        # acc_tt = (accuracy + truetrue)/2.0
        f1_socre = res['metrics']['F1']
        if f1_socre > self.top_f1:
            self.top_f1 = f1_socre
            save_tag = True
        else:
            save_tag = False
        # cp_sufix = 'accuracy_%.3f_tt_%.3f_.pkl' % (accuracy, truetrue)
        cp_sufix = 'f1_%.3f_.pkl' % f1_socre
        return save_tag, cp_sufix