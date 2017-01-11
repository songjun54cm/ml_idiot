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
        self.fold_splits = list()

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

    def get_n_fold_splits(self, sample_list, n):
        rate_list = [ 1.0/n for i in xrange(n) ]
        fold_splits = self.get_split_data(sample_list, rate_list)
        return fold_splits

    def form_splits(self, train_folds, train_valid_fold, valid_fold, test_fold):
        self.splits['train'] = []
        for fold_id in train_folds:
            self.splits['train'] += self.fold_splits[fold_id]
        self.splits['train_valid'] = self.fold_splits[train_valid_fold]
        self.splits['valid']= self.fold_splits[valid_fold]
        self.splits['test'] = self.fold_splits[test_fold]

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

    def get_split_fold_nums(self, fold_num, k):
        valid_fold = fold_num
        test_fold = (fold_num+1)%k
        train_valid_fold = (fold_num+2)%k
        train_folds = (np.arange(k-2) + fold_num + 2) % k
        return train_folds, train_valid_fold, valid_fold, test_fold

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

    def iter_training_batch(self, batch_size, rng=random.Random(1234)):
        for iter_data in self.iter_split_batches(batch_size, 'train', rng=rng):
            yield iter_data

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