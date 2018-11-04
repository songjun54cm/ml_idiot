__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.data_provider.NormalDataProvider import NormalDataProvider


class LRDataProvider(NormalDataProvider):
    def __init__(self):
        super(LRDataProvider, self).__init__()
        self.split_ratios = [0.8, 0.1, 0.1]

    def load_raw_data_samples(self, config):
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        records = []
        for (a, b) in zip(X, y):
            if b in [0,1]:
                records.append((a,b))
        return records

    def create(self, config):
        super(LRDataProvider, self).build(config)
        super(LRDataProvider, self).summarize()
