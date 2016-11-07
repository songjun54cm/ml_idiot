import pickle
import random
import numpy as np
__author__ = 'SongJun-Dell'

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

    def get_split(self, split):
        return self.splits[split]

    def split_size(self, split):
        return len(self.splits[split])

    def get_split_data(self, samples_list, rate_list):
        num_sample = len(samples_list)
        random_idx = range(0, num_sample)
        random.seed(0)
        random.shuffle(random_idx)
        split_lens = [ int(srate * num_sample) for srate in rate_list]
        split_idxes = list()
        split_data_list = list()
        spos = 0
        epos = 0
        for pos in split_lens:
            epos += pos
            split_idxes.append(random_idx[spos:epos])
            spos = epos

        for idxes in split_idxes:
            split_data_list.append([samples_list[di] for di in idxes])
        return split_data_list

    def save(self, file_path):
        print 'trying to save provider into %s' % file_path
        with open(file_path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
        # self.splits = d['splits']
        for key in self.__dict__.keys():
            self.__dict__[key] = d[key]

    # def iter_split_batches(self, batch_size, split):
    #     batch = list()
    #     datas = self.get_split(split)
    #     for d in datas:
    #         batch.append(d)
    #         if len(batch) >= batch_size:
    #             yield batch
    #             batch = list()
    #
    #     if len(batch) > 0:
    #         yield batch

    def iter_split_batches(self, batch_size, split, rng=random.Random(1234)):
        split_size = len(self.splits[split])
        idxs = range(split_size)
        rng.shuffle(idxs)
        split_datas = self.splits[split]
        start_pos = 0
        while start_pos < split_size:
            end_pos = start_pos + batch_size
            iter_datas = [split_datas[idxs[id]] for id in xrange(start_pos, min(split_size,end_pos))]
            start_pos = end_pos
            yield self.form_data(iter_datas)

    def form_data(self, batch_data, options=None):
        return batch_data