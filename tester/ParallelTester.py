__author__ = 'JunSong<songjun54cm@gmail.com>'
from multiprocessing import Pool, current_process
from tqdm import tqdm
import numpy as np
import logging

from ml_idiot.tester.BasicTester import BasicTester
from ml_idiot.utils.utils import get_data_splits

def get_pid(i):
    process = current_process()
    return process.pid

def load_model(m):
    global model
    model = m
    return True


def test_batch_data_v0(model, data_samples, ppid):
    res = {
        'loss': [],
        'pred_vals': [],
        'gth_vals': []
    }
    pid = current_process().pid
    if ppid == pid:
        # logging.info('ppid: %s, selected process' % str(ppid))
        for sample in tqdm(data_samples):
            loss, pred_val, gth_val = model.loss_one_sample_predict(sample)
            res['loss'].append(loss)
            res['pred_vals'].append(pred_val)
            res['gth_vals'].append(gth_val)
    else:
        # logging.info('ppid: %s, not selected process: %s' % (str(ppid), str(pid)))
        for sample in data_samples:
            loss, pred_val, gth_val = model.loss_one_sample_predict(sample)
            res['loss'].append(loss)
            res['pred_vals'].append(pred_val)
            res['gth_vals'].append(gth_val)
    return res


def test_batch_data(data_samples, ppid):
    global model
    res = {
        'loss': [],
        'pred_vals': [],
        'gth_vals': []
    }
    pid = current_process().pid
    if ppid == pid:
        # logging.info('ppid: %s, selected process' % str(ppid))
        for sample in tqdm(data_samples):
            loss, pred_val, gth_val = model.loss_one_sample_predict(sample)
            res['loss'].append(loss)
            res['pred_vals'].append(pred_val)
            res['gth_vals'].append(gth_val)
    else:
        # logging.info('ppid: %s, not selected process: %s' % (str(ppid), str(pid)))
        for sample in data_samples:
            loss, pred_val, gth_val = model.loss_one_sample_predict(sample)
            res['loss'].append(loss)
            res['pred_vals'].append(pred_val)
            res['gth_vals'].append(gth_val)
    return res

class ParallelTester(BasicTester):
    """
    test model using multi-processing
    will call:  loss, pred_val, gth_val = model.loss_one_sample_predict(one_sample)
    """
    def __init__(self, config):
        super(ParallelTester, self).__init__(config)
        self.num_processes = self.config.get('num_processes', 10)
        self.pool = Pool(processes=self.num_processes)
        logging.info('create %d process for testing.' % self.num_processes)
        self.process_ids = self.pool.map(get_pid, range(self.num_processes))
        self.model_loaded = False

    def pool_load_model(self, model):
        logging.info('load model for each process.')
        self.pool.map(load_model, [model for _ in range(self.num_processes)])
        self.model_loaded = True

    def test_on_split(self, model, data_provider, split):
        if not self.model_loaded:
            self.pool_load_model(model)
        ppid = self.process_ids[0]
        data_samples = data_provider.get_split(split)
        sample_num = len(data_samples)
        data_batches = get_data_splits(self.num_processes, data_samples)
        pool_results = []
        losses = []
        pred_vals = []
        gth_vals = []
        for pro in range(self.num_processes):
            samples = data_batches[pro]
            tmp_res = self.pool.apply_async(test_batch_data, (samples, ppid))
            pool_results.append(tmp_res)

        for pro in range(self.num_processes):
            pro_res = pool_results[pro].get()
            losses += pro_res['loss']
            pred_vals += pro_res['pred_vals']
            gth_vals += pro_res['gth_vals']

        res = {
            'loss': np.mean(losses),
            'pred_vals': pred_vals,
            'gth_vals': gth_vals,
            'sample_num': sample_num
        }
        return res

    def test_on_split_v0(self, model, data_provider, split):
        if not self.model_loaded:
            self.pool_load_model(model)
        ppid = self.process_ids[0]
        data_samples = data_provider.get_split(split)
        sample_num = len(data_samples)
        data_batches = get_data_splits(self.num_processes, data_samples)
        pool_results = []
        losses = []
        pred_vals = []
        gth_vals = []
        for pro in range(self.num_processes):
            samples = data_batches[pro]
            tmp_res = self.pool.apply_async(test_batch_data, (model, samples, ppid))
            pool_results.append(tmp_res)

        for pro in range(self.num_processes):
            pro_res = pool_results[pro].get()
            losses += pro_res['loss']
            pred_vals += pro_res['pred_vals']
            gth_vals += pro_res['gth_vals']

        res = {
            'loss': np.mean(losses),
            'pred_vals': pred_vals,
            'gth_vals': gth_vals,
            'sample_num': sample_num
        }
        return res