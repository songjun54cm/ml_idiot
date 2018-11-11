__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import numpy as np
from ml_idiot.data_provider.NormalDataProvider import NormalDataProvider


class SJLRDataProvider(NormalDataProvider):
    def __init__(self):
        super(SJLRDataProvider, self).__init__()
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
        super(SJLRDataProvider, self).build(config)
        super(SJLRDataProvider, self).summarize()

    def form_batch_data(self, samples, option=None):
        x = []
        y = []
        for(fea, label) in samples:
            x.append(fea)
            y.append(label)
        x = np.asarray(x, dtype=np.float32)
        y =  np.expand_dims(np.asarray(y, dtype=np.int), axis=1)
        batch_data = {
            "x": x,
            "y": y,
            "batch_size": x.shape[0]
        }
        return batch_data