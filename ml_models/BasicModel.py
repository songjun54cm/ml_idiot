__author__ = 'SongJun-Dell'
import logging
import _pickle as pkl
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
        with open(file_path, 'wb') as f:
            pkl.dump(self.__dict__, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            d = pkl.load(f)
        # self.splits = d['splits']
        for key in self.__dict__.keys():
            self.__dict__[key] = d[key]