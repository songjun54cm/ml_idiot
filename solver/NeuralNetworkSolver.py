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
        self.batch_size = state['batch_size']
        self.max_epoch = state['max_epoch']
        self.valid_epoch = state['valid_epoch']
        self.valid_batch_size = state['valid_batch_size']
        self.loss_scale = state['loss_scale']

    def setup_train_state(self, data_provider):
        # calculate how many iteration
        self.train_size = data_provider.split_size('train')
        self.valid_sample_num = max(1, int(self.train_size * self.valid_epoch))
        self.valid_sample_count = 0
        self.iter_count = 0
        self.valid_count = 0
        self.last_loss = np.inf
        self.top_valid_cost = np.inf
        self.smooth_valid_loss = dict()
        self.grad_cache = dict()
        self.last_save_model_file_path = None
        self.top_performance_csv_message = None
        self.top_performance_valid_res = None
        self.tester.init_tester()

    def train(self, model, data_provider):
        return self.normal_train(model, data_provider)

    def normal_train(self, model, data_provider):
        self.create_out_folder()
        self.create_log_files()
        self.create_checkpoint_dir()
        self.log_train_message(json.dumps(self.state, indent=2), file_name='model_state.txt')
        self.setup_train_state(data_provider)

        self.train_epoches(model ,data_provider)

        self.log_valid_csv_message(self.top_performance_csv_message, 'top_performance_log.csv')
        for res_name in ['train', 'valid', 'test']:
            message = self.form_valid_message(self.top_performance_valid_res[res_name])
            self.log_train_message(message)
        return self.last_save_model_file_path, self.top_performance_valid_res

    def train_epoches(self, model, data_provider):
        for epoch_i in xrange(self.max_epoch):
            self.sample_count = 0
            self.epoch_i = epoch_i
            for batch_data in data_provider.iter_training_batch(self.batch_size, self.rng):
                self.train_one_batch(model, batch_data, epoch_i)

                # validation
                if self.to_validate(epoch_i):
                    self.valid_sample_count = 0
                    self.valid_count += 1
                    valid_csv_message = 'epoch_num,%d\n' % epoch_i
                    valid_csv_message += self.form_valid_csv(mode='head') + '\n'
                    # validate on the train valid data set
                    train_res = self.validate_on_split(model, data_provider, split='train_valid')
                    valid_csv_message += self.form_valid_csv(mode='body', res=train_res) + '\n'
                    # validate on the validate data set
                    valid_res = self.validate_on_split(model, data_provider, split='valid')
                    valid_csv_message += self.form_valid_csv(mode='body', res=valid_res) + '\n'
                    # validate on the test data set
                    test_res = self.validate_on_split(model, data_provider, split='test')
                    valid_csv_message += self.form_valid_csv(mode='body', res=test_res) + '\n'
                    self.log_valid_csv_message(valid_csv_message)
                    self.log_valid_csv_message('\n')
                    validate_res = {'train': train_res, 'valid': valid_res, 'test': test_res}

                    self.save_or_not(valid_res, model, valid_csv_message, validate_res)
            self.decay_learning_rate()
    def decay_learning_rate(self):
        self.state['learning_rate'] = self.state['learning_rate'] * self.state['learning_rate_decay']

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
        csv_message = 'epoch,%.3f,/,%d,samples,%d,/,%d,done_in,%.3f,seconds,loss_cost,%f,smooth,%f' \
            % (epoch_rate, self.max_epoch, self.sample_count, self.train_size, time_eclipse, loss, self.smooth_train_cost)
        self.log_train_csv(csv_message)
        # detect loss exploding
        if not self.detect_loss_explosion(loss):
            sys.exit()

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

    def validate_on_split(self, model, data_provider, split):
        t0 = time.time()
        res = self.tester.test_on_split(model, data_provider, split)
        metrics = self.tester.get_metrics(res, self.metrics)
        time_eclipse = time.time() - t0

        results = dict()
        # results.update(res)
        results['metrics'] = metrics
        results['sample_num'] = res['sample_num']
        results['seconds'] = time_eclipse
        results['split'] = split
        results['loss'] = res['loss']*self.state['loss_scale']
        message = self.form_valid_message(results)
        self.log_train_message(message)
        return results

    def form_valid_message(self, res):
        message = ''
        for key in self.metrics:
            message += '%s: %.5f ' % (key, res['metrics'][key])
        message = 'evaluate %10d %15s samples in %.3fs, Loss: %5.3f. ' \
                  % (res['sample_num'], res['split'], res['seconds'], res['loss']) + message
        return message


    def form_valid_csv(self, mode, res=None):
        if mode == 'head':
            head_message = 'sample_num,seconds,split,Loss,'+','.join(self.metrics)
            return head_message
        elif mode == 'body':
            body_message = '%d,%.3f,%s,%f' % (res['sample_num'], res['seconds'], res['split'], res['loss'])
            for met in self.metrics:
                body_message += ',%f' % res['metrics'][met]
            return body_message
        else:
            raise StandardError('form_valid_csv mode error.')

    def test(self, model, data_provider):
        return self.normal_test(model, data_provider)

    def normal_test(self, model, data_provider, split=None):
        if split is None:
            train_res = self.validate_on_split(model, data_provider, 'train_valid')
            valid_res = self.validate_on_split(model, data_provider, 'valid')
            test_res = self.validate_on_split(model, data_provider, 'test')
            test_result = {'train': train_res, 'valid': valid_res, 'test': test_res}
        else:
            res = self.validate_on_split(model ,data_provider, split)
            test_result = {split: res}
        return test_result