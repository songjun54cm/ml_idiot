__author__ = 'SongJun-Dell'
from settings import ProjectHome
from ml_idiot.utils.utils import path_join

def get_data_folder(config):
    return path_join(config['proj_folder'], 'data', config['data_set_name'])


def complete_config(config):
    config['proj_folder'] = ProjectHome
    config['data_folder'] = get_data_folder(config)
    config['mode'] = config.get('mode', 'train')
    config['loss_scale'] = config.get('loss_scale', 1.0)
    config['data_provider'] = config.get('data_provider', '%sDataProvider' % config['data_set_name'])
    return config

