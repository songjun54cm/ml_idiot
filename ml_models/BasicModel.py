__author__ = 'SongJun-Dell'
import logging
from ml_idiot.utils.save_load import pkl_dump, pkl_load, save_dict, load_dict
import abc
import numpy as np


def add_to_params(params, param_val, param_name):
    params[param_name] = param_val
    return param_val


def init_matrix(shape, rng=np.random.RandomState(1234), name=None, magic_number=None):
    if magic_number is None:
        magic_number = 1.0 / np.power(np.prod(shape), 1.0/len(shape))
    return rng.standard_normal(shape) * magic_number


class BasicModel(object):
    def __init__(self, config):
        self.save_ext = 'pkl'
        self.config = config
        self.params = {}
        self.regularize_param_names = []

    def loss_batch_predict(self, batch_samples):
        """
        predict one batch data
        :param batch_samples:   list of samples
        :return:    loss, pred_vals, gth_vals
        """
        raise NotImplementedError

    def loss_one_sample_predict(self, sample):
        """
        predict one sample data with loss
        :param sample: one data sample
        :return:    loss, pred_val, gth_val
        """
        raise NotImplementedError

    def save(self, file_path):
        logging.info('trying to save model into %s' % file_path)
        pkl_dump(self.__dict__, file_path)

    def load(self, file_path):
        d = pkl_load(file_path)
        # self.splits = d['splits']
        for key in self.__dict__.keys():
            self.__dict__[key] = d[key]

    def save_to_dir(self, target_path):
        logging.info('trying to save model into %s' % target_path)
        save_dict(self.__dict__, target_path)

    def load_to_dir(self, source_path):
        dict_data = load_dict(source_path)
        for key in self.__dict__.keys():
            self.__dict__[key] = dict_data[key]

    def get_params(self):
        return self.params

    def add_params(self, shape, param_name, value_type="matrix"):
        if value_type == 'matrix':
            param_values = add_to_params(self.params, init_matrix(shape), param_name=param_name)
        elif value_type == 'bias':
            param_values = add_to_params(self.params, np.zeros(shape), param_name=param_name)
        else:
            raise BaseException('value type error.')
        return param_values, param_name

