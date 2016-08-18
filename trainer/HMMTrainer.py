__author__ = 'SongJun-Dell'
import numpy as np
import time


class HMMTrainer(object):
    def __init__(self):
        pass

    def train(self, model, data_provider):
        train_split = data_provider.splits['train']
        # print train_split.keys()
        # obs_seqs = train_split['obs_seqs']
        # state_seqs = train_split['state_seqs']

        t0 = time.time()
        train_res = model.fit(train_split)
        eclipse_time = time.time() - t0
        message = 'train model with %d sequences in %f seconds' % (train_res['sample_num'], eclipse_time)
        return message
