__author__ = 'SongJun-Dell'
from settings import PROJECT_HOME
from ml_idiot.utils.utils import path_join

def get_data_folder(config):
    return path_join(config['proj_folder'], 'data', config['data_set_name'])


def complete_config(config):
    config['proj_folder'] = PROJECT_HOME
    config['data_folder'] = get_data_folder(config)
    config['mode'] = config.get('mode', 'train')
    config['loss_scale'] = config.get('loss_scale', 1.0)
    return config