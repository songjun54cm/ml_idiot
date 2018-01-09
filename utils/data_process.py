__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import random

def get_split_data(samples_list, rate_list, seed=0):
    """
    split data in samples_list into several splits according to rate_list
    :param samples_list: [list of sample data]
    :param rate_list: [list of rate data]
    :return: [ list of splits, each splits is a list of sample data ]
    """
    num_sample = len(samples_list)
    random_idx = range(0, num_sample)
    random.seed(seed)
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

def get_n_fold_splits(sample_list, n, seed=0):
    """
    get n fold splits, sample data are randomly and equally distributed into n fold .
    :param sample_list:
    :param n:
    :return:
    """
    rate_list = [ 1.0/n for _ in xrange(n) ]
    fold_splits = get_split_data(sample_list, rate_list, seed)
    return fold_splits

def main(state):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    state = vars(args)
    main(state)