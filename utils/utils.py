__author__ = 'SongJun-Dell'
import time, os

path_join = os.path.join

def update_config(config, user_config):
    for key in user_config.keys():
        if user_config[key] is not None:
            config[key] = user_config[key]
    return config

def get_data_folder(state):
    return path_join(state['proj_folder'], 'data', state['data_set_name'])

def get_dp_file_path(state):
    return path_join(state['data_folder'], '%s_%s_data_provider.pkl'%(state['model_name'], state['data_set_name']))

def counting_time(func):
    def _deco(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        time_eclipse = time.time() - t0
        print 'run %s finish in %.3f seconds.' % (func.__name__, time_eclipse)
        return ret
    return _deco


def isWindows():
    import platform
    sysstr = platform.system()
    if sysstr == 'Windows':
        return True
    else:
        return False


def sort_dict(dict_count):
    dic = sorted(dict_count.iteritems(), key=lambda d:d[1], reverse=True)
    return dic


def insert_list_to_voca(in_list, voca=None):
    if voca is None:
        voca = {
            'w_to_ix': dict(),
            'ix_to_w': list(),
            'w_count': list(),
            'next_ix': 0
        }
    for key in in_list:
        voca['w_to_ix'][key] = voca['next_ix']
        voca['ix_to_w'].append(key)
        voca['w_count'].append(0)
        voca['next_ix'] += 1
    return voca


def get_voca_from_count(key_count, insert_list=list(), unknow=True):
    if unknow:
        in_list = ['UNKNOW'] + insert_list
    else:
        in_list = insert_list
    voca = insert_list_to_voca(in_list)
    for k,v in key_count:
        voca['w_to_ix'][k] = voca['next_ix']
        voca['ix_to_w'].append(k)
        voca['w_count'].append(v)
        voca['next_ix'] += 1
    return voca


def get_data_splits(num_splits, datas_list):
    data_splits = list()
    batch_size = len(datas_list)
    step = max(int(batch_size/num_splits), 1)
    idxs = range(0, batch_size, step)
    if batch_size % step == 0:
        idxs.append(batch_size)
    else:
        idxs[-1] = batch_size
    # print batch_size
    # print idxs
    for i in xrange(len(idxs)-1):
        data_splits.append(datas_list[idxs[i]:idxs[i+1]])
    return data_splits


def set_nn_state(state):
    # set output folder
    if not state.has_key('out_folder'):
        num = 1
        out_prefix = os.path.join(state['proj_folder'], 'output', state['model_name'], state['data_set_name'], 'out')
        state['out_prefix'] = out_prefix
        out_folder = os.path.join(out_prefix, str(num))
        while os.path.exists(out_folder):
            num += 1
            out_folder = os.path.join(out_prefix, str(num))
        os.makedirs(out_folder)
        state['out_folder'] = out_folder
        state['train_log_file'] = os.path.join(out_folder, 'train_log.log')
        state['valid_log_file'] = os.path.join(out_folder, 'valid_log.log')
        state['checkpoint_out_dir'] = os.path.join(out_folder, 'check_point')
        os.makedirs(state['checkpoint_out_dir'])
    return state

"""
def setup_ml_state(state):
    if not state.has_key('out_folder'):
        num = 1
        out_prefix = os.path.join(state['proj_folder'], 'output', state['model_name'], state['data_set_name'], 'out')
        state['out_prefix'] = out_prefix
        out_folder = os.path.join(out_prefix, str(num))
        while os.path.exists(out_folder):
            num += 1
            out_folder = os.path.join(out_prefix, str(num))
        os.makedirs(out_folder)
        state['out_folder'] = out_folder
        state['train_log_file'] = os.path.join(out_folder, 'train_log.log')
        state['train_log_csv_file'] = os.path.join(out_folder, 'train_log_csv.csv')
        state['valid_log_csv_file'] = os.path.join(out_folder, 'valid_log_csv.csv')
        state['checkpoint_out_dir'] = os.path.join(out_folder, 'check_point')
        os.makedirs(state['checkpoint_out_dir'])
    return state
"""

def strdecode(sentence):
    if not isinstance(sentence, unicode):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk', 'ignore')
    return sentence



def pickle_load(file_path, verbose=False):
    import cPickle as pickle
    if verbose:
        print('load data from %s, ...' % file_path),
        start_time = time.time()
    with open(file_path, 'rb') as f:
        res = pickle.load(f)
    if verbose:
        print('finish. in %f seconds' % (time.time()-start_time))
    return res
pk_load = pickle_load

def pickle_dump(obj, file_path, verbose=False):
    if verbose:
        print('save data to %s, ...' % file_path),
        start_time = time.time()
    import cPickle as pickle
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        print('finish. in %f seconds' % (time.time()-start_time))
pk_dump = pickle_dump
