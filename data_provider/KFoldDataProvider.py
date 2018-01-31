__author__ = 'JunSong<songjun54cm@gmail.com>'
import abc
import numpy as np

from ml_idiot.data_provider.NormalDataProvider import NormalDataProvider
from ml_idiot.utils.data_process import get_n_fold_splits


def get_split_fold_nums(fold_num, k):
    valid_fold = fold_num
    test_fold = (fold_num + 1) % k
    train_valid_fold = (fold_num + 2) % k
    train_folds = (np.arange(k - 2) + fold_num + 2) % k
    return train_folds, train_valid_fold, valid_fold, test_fold

class KFoldDataProvider(NormalDataProvider):
    """Provide Data in K fold splits.
    Attributes:
        fold_splits: K fold data splits, each fold contain same number of samples
    """
    folds = {}
    num_folds = 10

    def __init__(self, num_folds=10):
        super(KFoldDataProvider, self).__init__()
        self.num_folds = num_folds

    @abc.abstractmethod
    def load_raw_data_samples(self, config):
        """
        load raw data samples and form a data sample list
        :param config:  configuresfol
        :return:    list of data samples
        """
        raise NotImplementedError

    def build(self, config):
        """
        build data provider
        :param config:  configurations
        :return:    None
        """
        samples_list = self.load_raw_data_samples(config)
        # create K fold splits with uniform distribution
        self.fold_splits = get_n_fold_splits(samples_list, self.num_folds, seed=0)

    def form_splits(self, train_folds, train_valid_fold, valid_fold, test_fold):
        self.splits['train'] = []
        for fold_id in train_folds:
            self.splits['train'] += self.folds[fold_id]
        self.splits['train_valid'] = self.folds[train_valid_fold]
        self.splits['valid']= self.folds[valid_fold]
        self.splits['test'] = self.folds[test_fold]


