__author__ = 'SongJun-Dell'
import time
import numpy as np
import datetime
import logging
import importlib
from ml_idiot.evaluator.BasicEvaluator import BasicEvaluator
from ml_idiot.data_provider.BasicDataProvider import get_batch_size
from ml_idiot.solver.BasicSolver import form_name


def create_evaluator(config):
    logging.info('create evaluator: %s ...' % config['evaluator'])
    eval_cls_name = form_name(config['evaluator'], 'Evaluator')
    eval_cls = getattr(importlib.import_module('evaluators.%s' % eval_cls_name), eval_cls_name)
    metrics = config.get("metrics", None)
    eval = eval_cls()
    eval.init_metric(metrics)
    return eval


class BasicTester(object):
    """
    BasicTester:
    will call:  loss, pred_vals, gth_vals = model.loss_batch_predict(batch_samples)
    """
    def __init__(self, config):
        self.config = config
        self.evaluator = create_evaluator(config)

    def test(self, model, data_provider, split=None):
        if split is None:
            train_res = self.validate_on_split(model, data_provider, 'train_valid')
            valid_res = self.validate_on_split(model, data_provider, 'valid')
            test_res = self.validate_on_split(model, data_provider, 'test')
            test_result = {'train': train_res, 'valid': valid_res, 'test': test_res}
        else:
            res = self.validate_on_split(model, data_provider, split)
            test_result = {split: res}
        return test_result

    def validate_on_split(self, model, data_provider, split):
        t0 = time.time()
        res = self.valid_split_metrics(model, data_provider, split)
        time_eclipse = time.time() - t0

        results = dict()
        # results.update(res)
        results['metrics'] = res['metrics']
        results['sample_num'] = res['sample_num']
        results['seconds'] = time_eclipse
        results['split'] = split
        results['loss'] = res['loss']*self.config.get('loss_scale', 1.0)
        return results

    def valid_split_metrics(self, model, data_provider, split):
        res = self.test_on_split(model, data_provider, split)
        metrics = self.evaluator.evaluate(res['gth_vals'], res['pred_vals'])
        res['metrics'] = metrics
        return res

    def test_on_split(self, model, data_provider, split):
        total_loss = []
        sample_num = 0
        gth_vals = []
        pred_vals = []
        for batch_data in data_provider.iter_split_batches(data_provider.valid_batch_size, split):
            res = self.test_one_batch(model, batch_data)
            total_loss.append(res['loss'])
            sample_num += res['sample_num']
            if isinstance(res['gth_vals'], list):
                gth_vals += res['gth_vals']
                pred_vals += res['pred_vals']
            elif isinstance(res['gth_vals'], np.ndarray):
                gth_vals.append(res['gth_vals'])
                pred_vals.append(res['pred_vals'])
            else:
                raise NotImplementedError
        res = {
            'loss': np.mean(total_loss),
            'sample_num': sample_num,
        }
        if isinstance(gth_vals[0], np.ndarray):
            res.update({
            'gth_vals': np.concatenate(gth_vals),
            'pred_vals': np.concatenate(pred_vals)
            })
        else:
            res.update({
                'gth_vals': gth_vals,
                'pred_vals': pred_vals
            })
        return res

    def test_one_batch(self, model, batch_data):
        loss, pred_vals, gth_vals = model.loss_batch_predict(batch_data)
        res = {
            'loss': loss,
            'sample_num': get_batch_size(batch_data),
            'gth_vals': gth_vals,
            'pred_vals': pred_vals
        }
        return res
