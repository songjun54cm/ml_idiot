__author__ = 'SongJun-Dell'
import pickle
import random
import numpy as np
import time
import logging
import abc

from ml_idiot.utils.data_process import get_n_fold_splits
from ml_idiot.utils.utils import counting_time

def get_batch_size(batch_data):
    if isinstance(batch_data, list):
        return len(batch_data)
    elif isinstance(batch_data, dict):
        return batch_data['batch_size']
    elif isinstance(batch_data, np.ndarray):
        return batch_data.shape[0]
    else:
        raise BaseException('get batch size error!')


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

    def summarize(self):
        """
        logging data provider basic information
        :return:
        """
        logging.info('data provider fields: %s' % str(list(self.__dict__.keys())))

    def create(self, config):
        """
        load data provider from pkl file or build it if pkl file not exist.
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
            logging.info('loaded data provider from %s.' % dp_file_path)
            self.load(dp_file_path)
        else:
            self.build(config)
            self.save(dp_file_path)
            logging.info('build data provider and save into %s' % dp_file_path)

        self.summarize()

    @counting_time
    def save(self, file_path, verbose=False):
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
            # print(list(self.__dict__.keys()))
        if verbose:
            logging.info('finish in %.2f seconds.' % (time.time()-stime))

    @counting_time
    def load(self, file_path, mode='full', verbose=False):
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
            raise BaseException('%s mode not recognised.' % mode)
        self.prepare_data()
        if verbose:
            logging.info('finish in %.2f seconds.' % (time.time()-start))



    def prepare_data(self):
        # prepare data after load from file
        pass

    def form_batch_data(self, samples, options=None):
        return samples


if __name__ == "__main__":
    import argparse
    from ml_idiot.config import complete_config
    from ml_idiot.data_provider import pre_build_dp
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config_file', dest='config_file', type=str, default='config_file_name')
    args = parser.parse_args()
    config = vars(args)
    config = complete_config(config)
    pre_build_dp(config, BasicDataProvider)

