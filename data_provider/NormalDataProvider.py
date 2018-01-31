__author__ = 'JunSong<songjun54cm@gmail.com>'
import abc, random
from copy import deepcopy

from ml_idiot.data_provider.BasicDataProvider import BasicDataProvider
from ml_idiot.utils.data_process import get_split_data

class NormalDataProvider(BasicDataProvider):
    """Provide Data in Three splits: train/valid/test and train_valid.
    Attributes:
        splits: data splits, train/valid/test/tran_valid
        split_ratio: the ratio of split train/valid/test
    """
    splits = dict()
    split_ratios = []

    def __init__(self):
        super(NormalDataProvider, self).__init__()

    @abc.abstractmethod
    def load_raw_data_samples(self, config):
        """
        load raw data samples and form a data sample list
        :param config:  configures
        :return:    list of data samples
        """
        raise NotImplementedError

    def form_train_data(self, samples):
        """
        form data samples into train structure
        :param samples: list of data samples
        :return:    list of train structure data samples
        """
        return samples

    def form_test_data(self, samples):
        """
        form data samples into test structure
        :param samples:     list of data samples
        :return:    list of test structure data samples
        """
        return samples

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
            'train': self.form_train_data(train_data),
            'valid': self.form_test_data(valid_data),
            'test': self.form_test_data(test_data),
            'train_valid': self.form_test_data(train_valid)
        }

    def get_split(self, split):
        return self.splits[split]

    def split_size(self, split):
        if isinstance(self.splits[split], list):
            return len(self.splits[split])
        elif isinstance(self.splits[split], dict):
            return self.splits[split]['split_size']

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
