__author__ = 'SongJun-Dell'
import json
import os
import time
import sys
from BasicSolver import BasicSolver
from ml_idiot.evaluator.RetrievalEvaluator import RetrievalEvaluator
import cPickle as pickle
import numpy as np

class NeuralNetworkSolver(BasicSolver):
    def __init__(self, state):
        super(NeuralNetworkSolver, self).__init__(state)
        self.top_valid_cost = np.inf
        self.batch_size = state['batch_size']
        self.max_epoch = state['max_epoch']
        self.valid_epoch = state['valid_epoch']
        self.valid_batch_size = state['valid_batch_size']
        self.loss_scale = state['loss_scale']

        self.last_loss = np.inf
        self.top_valid_cost = np.inf
        self.valid_count = 0
        self.valid_sample_count = 0
        self.iter_count = 0
        self.smooth_valid_loss = dict()
        self.grad_cache = dict()
        # self.top_valid_mrr = 0
        # self.top_valid_mrank = np.inf

    def setup_train_state(self, data_provider):
        # calculate how many iteration
        self.train_size = data_provider.split_size('train')
        self.valid_sample_num = max(1, int(self.train_size * self.valid_epoch))
        self.valid_sample_count = 0
        self.iter_count = 0
        self.valid_count = 0
        # self.evaluator = RetrievalEvaluator()

    def train(self, model, data_provider):
        self.create_checkpoint_dir()
        self.log_train_message(json.dumps(self.state, indent=2))
        self.setup_train_state(data_provider)

        for epoch_i in xrange(self.max_epoch):
            self.sample_count = 0
            self.epoch_i = epoch_i
            for batch_data in data_provider.iter_training_batch(self.batch_size, self.rng):
                self.train_one_batch(model, batch_data, epoch_i)

                # validation
                if self.to_validate(epoch_i):
                    self.valid_sample_count = 0
                    self.valid_count += 1
                    # validate on the train valid data set
                    res = self.validate_on_split(model, data_provider, split='train_valid')

                    # validate on the validate data set
                    valid_res = self.validate_on_split(model, data_provider, split='valid')

                    # validate on the test data set
                    res = self.validate_on_split(model, data_provider, split='test')

                    self.save_or_not(valid_res, model)
        return self.last_save_model_file_path

    def update_smooth_train_loss(self, new_loss):
        # calculate smooth cost
        if self.iter_count == 1:
            self.smooth_train_cost = new_loss
        else:
            self.smooth_train_cost = 0.99 * self.smooth_train_cost + 0.01 * new_loss

    def update_model_one_batch(self, model, batch_data):
        loss, grad_params = model.train_one_batch(batch_data)
        self.update_model(model, grad_params, mode=self.state['sgd_mode'])
        return loss

    def train_one_batch(self, model, batch_data, epoch_i):
        batch_size = self.get_batch_size(batch_data)
        self.iter_count += 1
        t0 = time.time()

        loss = self.update_model_one_batch(model, batch_data)

        loss *= self.loss_scale
        self.valid_sample_count += batch_size
        self.update_smooth_train_loss(loss)
        # print message
        time_eclipse = time.time() - t0
        self.sample_count += batch_size

        epoch_rate = epoch_i + 1.0 * self.sample_count / self.train_size
        message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. loss_cost= %f, (smooth %f)' \
            % (self.sample_count, self.train_size, time_eclipse, epoch_rate, self.max_epoch, loss, self.smooth_train_cost)
        self.log_train_message(message)
        # detect loss exploding
        if not self.detect_loss_explosion(loss):
            sys.exit()

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def validate_on_split(self, model, data_provider, split):
        t0 = time.time()
        res = self.tester.test_on_split(model, data_provider, split)
        valid_num = res['sample_num']
        metrics = self.tester.get_metrics(res, self.metrics)
        message = ''
        for key,value in metrics.iteritems():
            message += '%s: %f ' % (key, value)
        time_eclipse = time.time() - t0
        message = 'evaluate %d %s samples in %.3fs. ' % (valid_num, split, time_eclipse) + message
        self.log_message(message)

        results = dict()
        # results.update(res)
        results.update(metrics)
        results['loss'] = res['loss']*self.state['loss_scale']
        return results

    def test(self, model, data_provider):
        self.validate_on_split(model, data_provider, 'train_valid')
        self.validate_on_split(model, data_provider, 'valid')
        self.validate_on_split(model, data_provider, 'test')