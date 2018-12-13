__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/12/13
import abc

from data_provider.BasicDataProvider import BasicDataProvider


class TFDatasetDataProvider(BasicDataProvider):
    def __init__(self):
        super(TFDatasetDataProvider, self).__init__()
        self.train_data_iter = None
        self.valid_data_iter = None
        self.test_data_iter = None

    @abc.abstractmethod
    def build(self, config):
        raise NotImplementError

    def iter_train_batches(self):
        pass

    def iter_split_batches(self, split):
        pass
