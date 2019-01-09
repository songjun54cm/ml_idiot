__author__ = 'JunSong<songjun54cm@gmail.com>'
import abc, random
from copy import deepcopy
import logging

from ml_idiot.data_provider.BasicDataProvider import BasicDataProvider
from ml_idiot.utils.data_process import get_split_data


class NormalDataProvider(BasicDataProvider):
    """Provide Data in Three splits: train/valid/test and train_valid.
    Attributes:
        splits: data splits, train/valid/test/tran_valid
        split_ratio: the ratio of split train/valid/test
    """
    def __init__(self):
        super(NormalDataProvider, self).__init__()
        self.splits = {}
        self.split_ratios = []
        self.valid_batch_size = None

    @abc.abstractmethod
    def load_raw_data_samples(self, config):
        """
        load raw data samples and form a data sample list
        :param config:  configures
        :return:    list of data samples
        """
        raise NotImplementedError

    def has_split(self, split):
        return split in self.splits

    def build(self, config):
        """
        build data provider
        :param config:  configurations
        :return:    None
        """
        samples_list = self.load_raw_data_samples(config)
        train_data, valid_data, test_data = get_split_data(samples_list, self.split_ratios)
        train_valid = deepcopy(random.sample(train_data, len(valid_data)))
        self.splits = {
            'train': train_data,
            'valid': valid_data,
            'test': test_data,
            'train_valid': train_valid
        }

    def get_split(self, split):
        return self.splits.get(split, None)

    def split_size(self, split):
        if split not in self.splits:
            return -1
        if isinstance(self.splits[split], list):
            return len(self.splits[split])
        elif isinstance(self.splits[split], dict):
            return self.splits[split]['split_size']

    def iter_train_batches(self, batch_size, rng=random.Random(1234), opts=None):
        for iter_data in self.iter_split_batches(batch_size, 'train', rng=rng, opts=opts):
            yield iter_data

    def get_one_batch(self, split_datas, split_size, idxs, start_pos, end_pos):
        return [split_datas[idxs[id]] for id in range(start_pos, min(split_size, end_pos))]

    def iter_split_batches(self, batch_size, split, rng=random.Random(1234), shuffle=True, mode='random', opts=None):
        split_size = self.split_size(split)
        idxs = list(range(split_size))
        if shuffle:
            rng.shuffle(idxs)
        split_datas = self.splits[split]
        if batch_size is None or batch_size <= 0:
            iter_datas = self.get_one_batch(split_datas, split_size, idxs, 0, len(idxs))
            yield self.form_batch_data(iter_datas, opts)
        else:
            if mode == 'ordered':
                start_pos = 0
                while start_pos < split_size:
                    end_pos = start_pos + batch_size
                    iter_datas = self.get_one_batch(split_datas, split_size, idxs, start_pos, end_pos)
                    start_pos = end_pos
                    yield self.form_batch_data(iter_datas, opts)
            elif mode == 'random':
                for i in range(0, split_size, batch_size):
                    pos = random.randint(0, split_size-batch_size)
                    iter_datas = self.get_one_batch(split_datas, split_size, idxs, pos, pos+batch_size)
                    yield self.form_batch_data(iter_datas, opts)

    def summarize(self):
        super(NormalDataProvider, self).summarize()
        logging.info('train data size: %d' % self.split_size('train'))
        logging.info('valid data size: %d' % self.split_size('valid'))
        logging.info('test data size: %d' % self.split_size('test'))

    def create(self, config):
        self.build(config)
        self.summarize()
