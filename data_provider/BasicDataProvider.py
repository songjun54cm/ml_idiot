__author__ = 'SongJun-Dell'
import pickle
import random
import numpy as np
import time
import logging

def get_prob(freq_list):
    probs = np.array(freq_list, dtype=np.float32)
    probs /= np.sum(probs)
    probs += 1e-20
    probs = np.log(probs)
    probs -= np.max(probs)
    return probs

class BasicDataProvider(object):
    def __init__(self):
        self.splits = dict()
        self.fold_splits = list()\

    def build(self, config):
        print('build data provider from raw data')
        raise NotImplementedError

    def create(self, config):
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

    def get_split(self, split):
        return self.splits[split]

    def split_size(self, split):
        if isinstance(self.splits[split], list):
            return len(self.splits[split])
        elif isinstance(self.splits[split], dict):
            return self.splits[split]['split_size']

    def form_splits(self, train_folds, train_valid_fold, valid_fold, test_fold):
        self.splits['train'] = []
        for fold_id in train_folds:
            self.splits['train'] += self.fold_splits[fold_id]
        self.splits['train_valid'] = self.fold_splits[train_valid_fold]
        self.splits['valid']= self.fold_splits[valid_fold]
        self.splits['test'] = self.fold_splits[test_fold]

    def save(self, file_path, verbose=True):
        if verbose:
            stime = time.time()
            print('trying to save provider into %s' % file_path),
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if verbose:
            print('finish in %.2f seconds.' % (time.time()-stime))

    def load(self, file_path, mode='full', verbose=True):
        """

        :param file_path:
        :param mode: full: fill in all the field;
                     restrict: only fill in the field initialised.
        :param verbose:
        :return:
        """
        if verbose:
            start = time.time()
            print('loading data provider...'),
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
            print('finish in %.2f seconds.' % (time.time()-start))

    def get_split_fold_nums(self, fold_num, k):
        valid_fold = fold_num
        test_fold = (fold_num+1)%k
        train_valid_fold = (fold_num+2)%k
        train_folds = (np.arange(k-2) + fold_num + 2) % k
        return train_folds, train_valid_fold, valid_fold, test_fold

    def iter_training_batch(self, batch_size, rng=random.Random(1234), opts=None):
        for iter_data in self.iter_split_batches(batch_size, 'train', rng=rng, opts=opts):
            yield iter_data

    def iter_split_batches(self, batch_size, split, rng=random.Random(1234), shuffle=False, mode='ordered', opts=None):
        split_size = self.split_size(split)
        idxs = range(split_size)
        if shuffle:
            rng.shuffle(idxs)
        split_datas = self.splits[split]
        if mode=='ordered':
            start_pos = 0
            while start_pos < split_size:
                end_pos = start_pos + batch_size
                iter_datas = [split_datas[idxs[id]] for id in xrange(start_pos, min(split_size,end_pos))]
                start_pos = end_pos
                yield self.form_data(iter_datas, opts)
        elif mode=='random':
            for i in xrange(0,split_size,batch_size):
                pos = random.randint(0, split_size-batch_size)
                iter_datas = [split_datas[idxs[idx]] for idx in xrange(pos, pos+batch_size)]
                yield self.form_data(iter_datas, opts)

    def prepare_data(self):
        # prepare data after load from file
        pass

    def form_data(self, batch_data, options=None):
        return batch_data