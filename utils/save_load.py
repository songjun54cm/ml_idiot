__author__ = 'JunSong<songjun54cm@gmail.com>'
import time, os
import logging
import sys

try:
    import cPickle as pkl
except ImportError:
    import _pickle as pkl


def pickle_load(file_path, verbose=False):
    if verbose:
        logging.info('load data from %s, ...' % file_path),
        start_time = time.time()
    with open(file_path, 'rb') as f:
        res = pkl.load(f)
    if verbose:
        logging.info('finish. in %f seconds' % (time.time()-start_time))
    return res


def pickle_dump(obj, file_path, verbose=False):
    if verbose:
        logging.info('save data to %s, ...' % file_path),
        start_time = time.time()
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f)
    if verbose:
        logging.info('finish. in %f seconds' % (time.time()-start_time))


pk_load = pickle_load
pkl_load = pickle_load
pk_dump = pickle_dump
pkl_dump = pickle_dump


def data_dump(data, target_path):
    if isinstance(data, dict):
        assert '.' not in target_path, 'target path should be a path for a directory.'
        f = open(os.path.join(target_path, 'info.txt'), 'w')
        other_data = {}
        for key, val in data.items():
            if sys.getsizeof(val) > 1024 * 1024:
                pkl_dump(val, os.path.join(target_path, '%s.pkl' % key))
                f.write('%s\n' % key)
            else:
                other_data[key] = val
        if len(other_data) > 0:
            pkl_dump(other_data, os.path.join(target_path, 'other_data.pkl'))
            f.write("other_data: [ %s ]\n" % ', '.join(list(other_data.keys())))
        f.close()
    else:
        assert target_path.split('.')[-1] == 'pkl', 'target path should be .pkl file path'
        pkl_dump(data, target_path)


def data_load(source_path):
    if os.path.isdir(source_path):
        with open(os.path.join(source_path, 'info.txt'), 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        dict_data = {}
        for key in lines:
            if not key.startswith("other_data: ["):
                key_data = pkl_load(os.path.join(source_path, '%s.pkl' % key))
                dict_data[key] = key_data
            else:
                key_data = pkl_load(os.path.join(source_path, 'other_data.pkl'))
                dict_data.update(key_data)
        return dict_data
    elif os.path.isfile(source_path):
        assert source_path.split('.')[-1] == 'pkl', 'source path should be .pkl file path'
        return pkl_load(source_path)


def save_dict(dict_data, target_path):
    assert '.' not in target_path, 'target path should be a path for a directory.'
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    with open(os.path.join(target_path, 'info.txt'), 'w') as f:
        for key in dict_data.keys():
            f.write('%s\n' % key)
            pkl_dump(dict_data[key], os.path.join(target_path, '%s.pkl' % key))


def load_dict(source_path):
    assert '.' not in source_path, 'target path should be a path for a directory.'
    with open(os.path.join(source_path, 'info.txt'), 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    dict_data = {}
    for key in lines:
        dict_data[key] = pkl_load(os.path.join(source_path, '%s.pkl' % key))
    return dict_data

