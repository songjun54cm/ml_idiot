__author__ = 'SongJun-Dell'
import pickle
import random
import numpy as np
import time
import logging
import abc

from ml_idiot.utils.data_process import get_n_fold_splits

def get_prob(freq_list):
    probs = np.array(freq_list, dtype=np.float32)
    probs /= np.sum(probs)
    probs += 1e-20
    probs = np.log(probs)
    probs -= np.max(probs)
    return probs

class BasicDataProvider(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_raw_data_samples(self, config):
        """
        load raw data samples and form a data sample list
        :param config:  configures
        :return:    list of data samples
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build(self, config):
        """
        build data provider
        :param config:  configurations
        :return:    None
        """
        raise NotImplementedError

    def create(self, config):
        """
        load data provider from pkl file for build it if pkl file not exist.
        :param config:  configurations
        :return: None
        """
        import os
        from ml_idiot.data_provider import get_dp_file_path
        if ('dp_file' in config) and config['dp_file'] is not None :
            dp_file_path = config['dp_file']
        else:
            dp_file_path = get_dp_file_path(config)
        if os.path.exists(dp_file_path):
            print('start loading data provider from %s.' % dp_file_path)
            self.load(dp_file_path, verbose=True)
            logging.info('loaded data provider.')
        else:
            self.build(config)
            self.save(dp_file_path)
            logging.info('build data provider and save into %s' % dp_file_path)

    def save(self, file_path, verbose=True):
        """
        save data provider to pkl file
        :param file_path:   pkl file path
        :param verbose:     logging information or not
        :return:    None
        """
        stime = time.time()
        if verbose:
            logging.info('trying to save provider into %s' % file_path),
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if verbose:
            logging.info('finish in %.2f seconds.' % (time.time()-stime))

    def load(self, file_path, mode='full', verbose=True):
        """
        load data provider from pkl file
        :param file_path:   data provider pkl file path
        :param mode: full: fill in all the field;
                     restrict: only fill in the field initialised.
        :param verbose: logging information or not
        :return:    None
        """
        if verbose:
            start = time.time()
            logging.info('loading data provider...'),
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
        # self.splits = d['splits']
        if mode=='restrict':
            for key in self.__dict__.keys():
                self.__dict__[key] = d[key]
        elif mode=='full':
            for key in d.keys():
                self.__dict__[key] = d[key]
        else:
            raise StandardError('%s mode not recognised.'%mode)
        self.prepare_data()
        if verbose:
            logging.info('finish in %.2f seconds.' % (time.time()-start))



    def prepare_data(self):
        # prepare data after load from file
        pass

    def form_data(self, batch_data, options=None):
        return batch_data