__author__ = 'SongJun-Dell'
import numpy as np
import logging

from ml_idiot.tester.BasicTester import BasicTester
from ml_idiot.evaluator.MultiLabelEvaluator import MultiLabelEvaluator


class MultiLabelTester(BasicTester):
    def __init__(self, config):
        super(MultiLabelTester, self).__init__(config)
        self.evaluator = MultiLabelEvaluator()
        self.init_tester()
        self.top_f1 = 0

    def init_tester(self):
        self.top_f1 = 0

    def test_batch(self, model, batch_data):
        gth_vals = batch_data['gth_vals']
        pred_vals, batch_loss = model.predict_batch(batch_data, with_loss=True)
        res = {
            'loss': batch_loss,
            'sample_num': batch_data['sample_num'],
            'gth_vals': gth_vals,
            'pred_vals': pred_vals,
        }
        return res

    def test_multiple_samples(self, model, sample_datas):
        gth_feas = [data['target_feas'] for data in sample_datas]
        preds = model.predict_multiple_samples(sample_datas)
        losses = list()
        for gfea, pfea in zip(gth_feas, preds):
            losses.append(model.loss_one_sample(gfea, pfea))
        loss = np.mean(losses)

        res = {
            'loss': loss,
            'sample_num': len(sample_datas),
            'gth_feas': np.concatenate(gth_feas),
            'pred_feas': np.concatenate(preds) if type(preds) is list else preds
        }
        return res

    def get_metric_value(self, res, met):
        gth_state_feas = res['gth_feas']
        pred_state_feas = res['pred_feas']
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
            logging.info('metric: %s' % met)
            raise BaseException('metric name error!')

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