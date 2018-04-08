__author__ = 'SongJun-Dell'
import logging
from utils.save_load import pkl_dump, pkl_load, save_dict, load_dict
import abc

class BasicModel(object):
    def __init__(self, config):
        self.save_ext = 'pkl'
        self.config = config

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

