__author__ = 'SongJun-Dell'
import json
import os
import time
import sys
from BasicSolver import BasicSolver
from ml_idiot.evaluator.RetrievalEvaluator import RetrievalEvaluator
import cPickle as pickle
import numpy as np
from copy import deepcopy
from ml_idiot.solver.BasicSolver import log_to_file
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

    def train(self, model, data_provider, method='normal'):
        if method == 'normal':
            return self.normal_train(model, data_provider)
        elif method == 'k_fold':
            return self.k_fold_train(model, data_provider)
        else:
            raise StandardError('train method error.')

    def normal_train(self, model, data_provider):
        self.create_out_folder()
        self.create_log_files()
        self.create_checkpoint_dir()
        self.log_train_message(json.dumps(self.config, indent=2), file_name='model_config.txt')
        self.setup_train_state(data_provider)

        self.train_epoches(model ,data_provider)

        self.log_valid_csv_message(self.top_performance_csv_message, 'top_performance_log.csv')
        for res_name in ['train', 'valid', 'test']:
            message = self.form_valid_message(self.top_performance_valid_res[res_name])
            self.log_train_message(message)
        return self.last_save_model_file_path, self.top_performance_valid_res

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
        results['loss'] = res['loss']*self.config['loss_scale']
        message = self.form_valid_message(results)
        self.log_train_message(message)
        return results

    def valid_split_metrics(self, model, data_provider, split):
        res = self.test_on_split(model, data_provider, split)
        metrics = self.tester.get_metrics(res, self.metrics)
        res['metrics'] = metrics
        return res

    def test_on_split(self, model, data_provider, split):
        total_loss = []
        sample_num = 0
        gth_feas = []
        pred_feas = []
        for batch_data in data_provider.iter_split_batches(self.valid_batch_size, split):
            res = self.test_one_batch(model, batch_data)
            total_loss.append(res['loss'])
            sample_num += res['sample_num']
            gth_feas.append(res['gth_feas'])
            pred_feas.append(res['pred_feas'])
        res = {
            'loss': np.mean(total_loss),
            'sample_num': sample_num,
            'gth_feas': np.concatenate(gth_feas),
            'pred_feas': np.concatenate(pred_feas)
        }
        return res

    def test_one_batch(self, model, batch_data):
        outs = model.loss_pred_on_batch(batch_data['x'], batch_data['y'])
        loss = outs[0]
        preds = outs[1]
        gth_feas = batch_data['y']
        res = {
            'loss': loss,
            'sample_num': self.get_batch_size(batch_data),
            'gth_feas': np.concatenate(gth_feas) if isinstance(gth_feas, list) else gth_feas,
            'pred_feas': np.concatenate(preds) if isinstance(preds, list) else preds
        }
        return res

    def test(self, model, data_provider, method='normal'):
        if method == 'normal':
            return self.normal_test(model, data_provider)
        elif method == 'k_fold':
            return self.k_fold_test(model, data_provider)
        else:
            raise StandardError('test method error.')

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

    def k_fold_train(self, model, data_provider):
        self.create_out_folder()
        self.config['exp_out_folder'] = self.config['out_folder']
        k = len(data_provider.fold_splits)
        orig_state = deepcopy(self.config)
        folds_validate_res = list()
        for fold_num in xrange(k):
            self.config = deepcopy(orig_state)
            train_folds, train_valid_fold, valid_fold, test_fold = data_provider.get_split_fold_nums(fold_num, k)
            data_provider.form_splits(train_folds, train_valid_fold, valid_fold, test_fold)
            self.config['out_folder'] = os.path.join(self.config['exp_out_folder'], 'fold_%d'%fold_num)
            self.config['train_folds'] = list(train_folds)
            self.config['train_valid_fold'] = train_valid_fold
            self.config['valid_fold'] = valid_fold
            self.config['test_fold'] = test_fold
            model.reinit()
            cp_path, validate_res = self.normal_train(model, data_provider)
            folds_validate_res.append(validate_res)
            message = 'save model check point into %s' % cp_path
            self.log_train_message(message)
            for res_name in ['train', 'valid', 'test']:
                message = self.form_valid_message(validate_res[res_name])
                self.log_train_message(message)
        print '\n'
        print '='*20
        message = '%d-Fold validation mean performance.' % k
        print message
        csv_log_file = '%d_fold_valid_log.csv' % k
        log_to_file(message, csv_log_file, self.config['exp_out_folder'])
        log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['exp_out_folder'])
        mean_result = {}
        for res_name in ['train', 'valid', 'test']:
            res_value = self.mean_res(folds_validate_res, res_name)
            mean_result[res_name] = res_value
            message = self.form_valid_message(res_value)
            print message
            csv_message = self.form_valid_csv(mode='body', res=res_value)
            log_to_file(csv_message, csv_log_file, self.config['exp_out_folder'])

        print '%d fold training and validation finish.' % k
        return self.last_save_model_file_path, mean_result

    def mean_res(self, res_list, key):
        key_res = dict()
        key_res['split'] = key
        for k in res_list[0][key].keys():
            if k not in ['metrics', 'split']:
                key_res[k] = np.mean([res[key][k] for res in res_list])
            elif k == 'metrics':
                key_res['metrics'] = {}
                for km in res_list[0][key]['metrics'].keys():
                    key_res['metrics'][km] = np.mean([res[key]['metrics'][km] for res in res_list])
        return key_res

    def k_fold_test(self, model, data_provider):
        self.create_out_folder()
        k = len(data_provider.fold_splits)
        csv_log_file = '%d_fold_test_log.csv' % k
        folds_validate_res = list()
        for fold_num in xrange(k):
            valid_fold = fold_num
            test_fold = (fold_num+1)%k
            train_valid_fold = (fold_num+2)%k
            train_folds = (np.arange(k-2) + fold_num + 2) % k
            data_provider.form_splits(train_folds, train_valid_fold, valid_fold, test_fold)
            print 'test %d fold' % test_fold
            test_result = self.normal_test(model, data_provider)
            folds_validate_res.append(test_result)
            csv_message = 'test fold,%d' % k
            log_to_file(csv_message, csv_log_file, self.config['out_folder'])
            log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['out_folder'])
            for res_name in ['train', 'valid', 'test']:
                # print self.form_valid_message(test_result[res_name])
                # print '\n'
                csv_message = self.form_valid_csv(mode='body', res=test_result[res_name])
                log_to_file(csv_message, csv_log_file, self.config['out_folder'])
            log_to_file('\n', csv_log_file, self.config['out_folder'])
        print '='*20
        message =  '%d-Fold validation mean performance.' % k
        print message
        log_to_file(message, csv_log_file, self.config['out_folder'])
        log_to_file(self.form_valid_csv(mode='head'), csv_log_file, self.config['out_folder'])
        for res_name in ['train', 'valid', 'test']:
            res_value = self.mean_res(folds_validate_res, res_name)
            print self.form_valid_message(res_value)
            log_to_file(self.form_valid_csv(mode='body', res=res_value), csv_log_file, self.config['out_folder'])
        print '%d fold test finish.' % k
        print 'test results have been saved into %s' % os.path.join(self.config['out_folder'], csv_log_file)