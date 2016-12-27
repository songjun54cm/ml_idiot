__author__ = 'SongJun-Dell'
import os, json, time, sys, random
import numpy as np
import cPickle as pickle
from ml_idiot.tester.BasicTester import BasicTester


def add_new_line(message):
    if message[-1] != '\n': message += '\n'
    return message

def log_to_file(message, file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'ab') as f:
        f.write(message)

class BasicSolver(object):
    def __init__(self, state):
        self.state = state
        self.tester = BasicTester()
        self.rng = random.Random(1234)
        self.train_log_file = None
        self.valid_log_file = None
        self.grad_cache = dict()
        self.metrics = []
        if state['mode'] == 'train':
            num = 1
            while os.path.exists(state['train_log_file']):
                state['train_log_file'] = state['train_log_file'] + '_' + str(num)
                num += 1
            while os.path.exists(state['valid_log_file']):
                state['valid_log_csv_file'] = state['valid_log_csv_file'] + '_' + str(num)
                num += 1
            while os.path.exists(state['train_log_csv_file']):
                state['train_log_csv_file'] = state['train_log_csv_file'] + '_' + str(num)
                num += 1
            self.train_log_file = open(state['train_log_file'], 'ab')
            self.train_log_csv_file = open(state['train_log_csv_file'], 'ab')
            self.valid_log_csv_file = open(state['valid_log_csv_file'], 'ab')
        self.last_save_model_file_path = None
        self.top_performance_csv_message = None

    def log_train_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        if self.train_log_file is not None:
            self.train_log_file.write(message)
            self.train_log_file.flush()
        if file_name is not None:
            file_path = os.path.join(self.state['out_folder'], file_name)
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
            log_to_file(message, file_name, self.state['out_folder'])

    def log_message(self, message, file_name=None):
        print message
        if message[-1] != '\n': message += '\n'
        self.log_train_message(message)
        self.log_valid_message(message)
        if file_name is not None:
            log_to_file(message, file_name, self.state['out_folder'])

    def detect_loss_explosion(self, loss):
        if loss > self.smooth_train_cost * 100:
            message = 'Aborting, loss seems to exploding. try to run gradient check or lower the learning rate.'
            self.log_train_message(message)
            return False
        # self.smooth_train_cost = loss
        return True

    def update_model(self, model, grad_params, mode='rmsprop'):
        grad_clip = self.state['grad_clip']
        momentum = self.state['momentum']
        decay_rate = self.state['decay_rate']
        smooth_eps = self.state['smooth_eps']
        learning_rate = self.state['learning_rate']

        if mode=='rmsprop':
            for p in grad_params.keys():
                grad_p = grad_params[p]
                # clip gradient
                if grad_clip > 0:
                    grad_p = np.minimum(grad_p, grad_clip)
                    grad_p = np.maximum(grad_p, -grad_clip)
                grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
                grad_p_cache = grad_p_cache * decay_rate + (1.0-decay_rate) * grad_p ** 2
                dx_p = -(learning_rate*grad_p)/np.sqrt(grad_p_cache+smooth_eps)

                self.grad_cache[p] = grad_p_cache
                model.params[p] += dx_p

        elif mode=='sgd':
            for p in grad_params.keys():
                grad_p = grad_params[p]
                if grad_clip > 0:
                    grad_p = np.minimum(grad_p, grad_clip)
                    grad_p = np.maximum(grad_p, -grad_clip)
                grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
                grad_p_cache = momentum * grad_p_cache + learning_rate * grad_p
                dx_p = -grad_p_cache

                self.grad_cache[p] = grad_p_cache
                model.params[p] += dx_p
        else:
            raise StandardError('sgd mode error!')

    def create_checkpoint_dir(self):
        if not os.path.exists(self.state['checkpoint_out_dir']):
            message = 'creating folder %s' % self.state['checkpoint_out_dir']
            self.log_train_message(message)
            os.makedirs(self.state['checkpoint_out_dir'])

    def save_or_not(self, res, model, valid_csv_message=None):
        save_tag, cp_suffix = self.tester.detect_to_save(res, model)
        modelcp_prefix = 'model_checkpoint_%s_%s' % (self.state['model_name'], self.state['data_set_name'])
        model_file_name = '%s_%s' % (modelcp_prefix, cp_suffix)
        # print save_tag
        if save_tag:
            model_file_path = os.path.join(self.state['checkpoint_out_dir'], model_file_name)
            model.save(model_file_path)
            message = 'save checkpoint models in %s.' % model_file_path
            self.log_train_message(message)
            self.log_valid_csv_message(message)
            self.log_valid_message('\n')
            self.last_save_model_file_path = model_file_path
            if valid_csv_message is not None:
                self.top_performance_csv_message = valid_csv_message

    def get_batch_size(self, batch_data):
        if type(batch_data)==list:
            return len(batch_data)
        else:
            raise StandardError('get batch size error!')

    def reform_dp(self, data_provider):
        pass