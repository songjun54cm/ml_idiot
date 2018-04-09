__author__ = 'SongJun-Dell'
import json
import os
import time
import sys
from trainers.BasicSolver_Trainer import BasicSolver
from ml_idiot.evaluator.RetrievalEvaluator import RetrievalEvaluator
import cPickle as pickle
import numpy as np
from ml_idiot.trainer.BasicSolver_Trainer import log_to_file
import logging

"""
config = {
    'batch_size': 0,
    'max_epoch': 0,
    'valid_epoch': 0,
    'valid_batch_size': 0,
    'loss_scale': 1,
}
"""

class NeuralNetworkSolver(BasicSolver):
    def __init__(self, config):
        super(NeuralNetworkSolver, self).__init__(config)
        self.batch_size = config['batch_size']
        self.max_epoch = config['max_epoch']
        self.valid_batch_size = config['valid_batch_size']
        self.loss_scale = config['loss_scale']
        self.learning_rate_decay = config.get('learning_rate_decay', 1.0)

        self.valid_epoch = config.get('valid_epoch', None)
        self.valid_iter = config.get('valid_iter', None)

    def setup_train_state(self, data_provider):
        # calculate how many iteration
        self.train_size = data_provider.split_size('train')

        v_num1 = int(self.train_size * self.valid_epoch) if self.valid_epoch is not None else self.train_size
        v_num2 = int(self.batch_size * self.valid_iter) if self.valid_iter is not None else self.train_size
        self.valid_sample_num = min(v_num1, v_num2)
        logging.info('valid sample number: %d'%self.valid_sample_num)

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
        self.top_metric = 0
        self.tester.init_tester()

    def train_model(self, model, data_provider):
        self.setup_train_state(data_provider)
        self.train_epoches(model, data_provider)

    def train_epoches(self, model, data_provider):
        self.valid_data(model, data_provider, 0)
        for epoch_i in xrange(self.max_epoch):
            self.sample_count = 0
            self.epoch_i = epoch_i
            for batch_data in data_provider.iter_training_batch(self.batch_size, self.rng):
                self.train_one_batch(model, batch_data, epoch_i)
                # validation
                if self.to_validate(epoch_i):
                    self.valid_sample_count = 0
                    self.valid_count += 1
                    train_res, valid_res, test_res, valid_csv_message = self.valid_data(model, data_provider, epoch_i)
                    validate_res = {'train': train_res, 'valid': valid_res, 'test': test_res}

                    self.save_or_not(valid_res, model, valid_csv_message, validate_res)
            self.decay_learning_rate()

    def valid_data(self, model, data_provider, epoch_i):
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
        return train_res, valid_res, test_res, valid_csv_message

    def decay_learning_rate(self):
        self.optimizer.decay_learning_rate()

    def update_model_one_batch(self, model, batch_data):
        loss, grad_params = model.train_one_batch(batch_data)
        self.optimizer.optimize_model(model, grad_params)
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
        message = 'samples %d/%d done in %.3fs. epoch %.3f/%d. loss= %f, (smooth %f)' \
            % (self.sample_count, self.train_size, time_eclipse, epoch_rate, self.max_epoch, loss, self.smooth_train_loss)
        self.log_train_message(message)
        csv_message = 'epoch,%.3f,/,%d,samples,%d,/,%d,done_in,%.3f,seconds,loss,%f,smooth,%f' \
            % (epoch_rate, self.max_epoch, self.sample_count, self.train_size, time_eclipse, loss, self.smooth_train_loss)
        self.log_train_csv(csv_message)
        # detect loss exploding
        if not self.detect_loss_explosion(loss):
            sys.exit()

    def update_smooth_train_loss(self, new_loss):
        # calculate smooth loss
        if self.iter_count == 1:
            self.smooth_train_loss = new_loss
            self.smooth_rate = 1.0 - min(0.01, (self.batch_size*1.0)/(self.train_size*1.0))
            logging.info('loss smooth rate: %f' % self.smooth_rate)

        else:
            self.smooth_train_loss = self.smooth_rate * self.smooth_train_loss + (1.0 - self.smooth_rate) * new_loss
