__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/7
import argparse
from data_provider.NormalDataProvider import NormalDataProvider


class LgbmGbdtDataProvider(NormalDataProvider):
    def __init__(self):
        super(LgbmGbdtDataProvider, self).__init__()
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
        super(LgbmGbdtDataProvider, self).build(config)
        super(LgbmGbdtDataProvider, self).summarize()
