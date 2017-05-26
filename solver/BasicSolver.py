__author__ = 'SongJun-Dell'
import os, json, time, sys, random
import numpy as np
import cPickle as pickle
from ml_idiot.tester.BasicTester import BasicTester
import importlib

def add_new_line(message):
    if message[-1] != '\n': message += '\n'
    return message

def log_to_file(message, file_name, folder_path):
    message = add_new_line(message)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'ab') as f:
        f.write(message)

def get_optimizer(config):
    from ml_idiot import optimizer
    opti = getattr(optimizer, config.get('optimizer', 'sgd'))(config)
    return opti

class BasicSolver(object):
    def __init__(self, config):
        self.config = config
        self.optimizer = get_optimizer(config)
        self.tester = BasicTester()
        self.rng = random.Random(1234)
        self.train_log_file = None
        self.valid_log_file = None
        self.grad_cache = dict()
        self.metrics = []

        self.last_save_model_file_path = None
        self.top_performance_csv_message = None

        self.top_metric = 0
        self.top_metric_name = ''

    def create_out_folder(self):
        if not self.config.has_key('out_folder'):
            num = 1
            out_prefix = os.path.join(self.config['proj_folder'], 'output',
                                      self.config['model_name'], self.config['data_set_name'], 'out')
            self.config['out_prefix'] = out_prefix
            out_folder = os.path.join(out_prefix, str(num))
            while os.path.exists(out_folder):
                num += 1
                out_folder = os.path.join(out_prefix, str(num))
            self.config['out_folder'] = out_folder
        if not os.path.exists(self.config['out_folder']):
            print 'create out folder %s.' % self.config['out_folder']
            os.makedirs(self.config['out_folder'])

    def create_log_files(self):
        self.config['train_log_file'] = os.path.join(self.config['out_folder'], 'train_log.log')
        self.config['train_log_csv_file'] = os.path.join(self.config['out_folder'], 'train_log_csv.csv')
        self.config['valid_log_csv_file'] = os.path.join(self.config['out_folder'], 'valid_log_csv.csv')
        if self.config['mode'] == 'train':
            num = 1
            while os.path.exists(self.config['train_log_file']):
                self.config['train_log_file'] = self.config['train_log_file'] + '_' + str(num)
                num += 1
            while os.path.exists(self.config['valid_log_csv_file']):
                self.config['valid_log_csv_file'] = self.config['valid_log_csv_file'] + '_' + str(num)
                num += 1
            while os.path.exists(self.config['train_log_csv_file']):
                self.config['train_log_csv_file'] = self.config['train_log_csv_file'] + '_' + str(num)
                num += 1
            self.train_log_file = open(self.config['train_log_file'], 'ab')
            self.train_log_csv_file = open(self.config['train_log_csv_file'], 'ab')
            self.valid_log_csv_file = open(self.config['valid_log_csv_file'], 'ab')

    def log_train_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        if self.train_log_file is not None:
            self.train_log_file.write(message)
            self.train_log_file.flush()
        if file_name is not None:
            file_path = os.path.join(self.config['out_folder'], file_name)
            with open(file_path, 'ab') as f:
                f.write(message)

    def log_train_csv(self, csv_message):
        csv_message = add_new_line(csv_message)
        if self.train_log_csv_file is not None:
            self.train_log_csv_file.write(csv_message)
            self.train_log_csv_file.flush()

    def log_valid_message(self, message):
        print message
        message = add_new_line(message)
        if self.valid_log_file is not None:
            self.valid_log_file.write(message)
            self.valid_log_file.flush()

    def log_valid_csv_message(self, message, file_name=None):
        csv_message = add_new_line(message)
        if self.valid_log_csv_file is not None:
            self.valid_log_csv_file.write(csv_message)
            self.valid_log_csv_file.flush()
        if file_name is not None:
            log_to_file(message, file_name, self.config['out_folder'])

    def log_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        self.log_train_message(message)
        self.log_valid_message(message)
        if file_name is not None:
            log_to_file(message, file_name, self.config['out_folder'])

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_cost * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True

    def create_checkpoint_dir(self):
        self.config['checkpoint_out_dir'] = os.path.join(self.config['out_folder'], 'check_point')
        if not os.path.exists(self.config['checkpoint_out_dir']):
            message = 'creating folder %s' % self.config['checkpoint_out_dir']
            self.log_train_message(message)
            os.makedirs(self.config['checkpoint_out_dir'])

    def save_or_not(self, res, model, valid_csv_message=None, validate_res=None):
        save_tag, cp_suffix = self.detect_to_save(res, model)
        modelcp_prefix = 'model_checkpoint_%s_%s' % (self.config['model_name'], self.config['data_set_name'])
        model_file_name = '%s_%s' % (modelcp_prefix, cp_suffix)
        # print save_tag
        if save_tag:
            model_file_path = os.path.join(self.config['checkpoint_out_dir'], model_file_name)
            model.save(model_file_path)
            message = 'save checkpoint models in %s.' % model_file_path
            self.log_train_message(message)
            self.log_valid_csv_message(message)
            self.log_valid_csv_message('\n')
            self.last_save_model_file_path = model_file_path
            if valid_csv_message is not None:
                self.top_performance_csv_message = valid_csv_message
            if validate_res is not None:
                self.top_performance_valid_res = validate_res

    def detect_to_save(self, res, model):
        metric_score = res['metrics'].get(self.top_metric_name)
        if metric_score > self.top_metric:
            self.top_metric = metric_score
            save_tag = True
        else:
            save_tag = False
        # cp_sufix = 'accuracy_%.3f_tt_%.3f_.pkl' % (accuracy, truetrue)
        cp_sufix = '%s_%.6f.%s' % (self.top_metric_name, metric_score, model.save_ext)
        return save_tag, cp_sufix

    def get_batch_size(self, batch_data):
        if isinstance(batch_data, list):
            return len(batch_data)
        elif isinstance(batch_data, dict):
            return batch_data['batch_size']
        elif isinstance(batch_data, np.ndarray):
            return batch_data.shape[0]
        else:
            raise StandardError('get batch size error!')

    def to_validate(self, epoch_i):
        return (self.valid_sample_count >= self.valid_sample_num) or \
               (epoch_i>=self.max_epoch-1 and self.sample_count>=self.train_size)

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


    def reform_dp(self, data_provider):
        pass