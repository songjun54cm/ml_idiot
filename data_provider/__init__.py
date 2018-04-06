__author__ = 'SongJun-Dell'
import logging

from ml_idiot.utils.utils import path_join
from ml_idiot.solver.BasicSolver import form_name


def get_dp_file_path(config):
    data_provider_name = form_name(config['data_provider'], 'DataProvider')
    return path_join(config['data_folder'], '%s.pkl'%(data_provider_name))

def pre_build_dp(config, dp_class):
    from ml_idiot.data_provider import get_dp_file_path
    import os
    dp_file_path = get_dp_file_path(config)
    if os.path.exists(dp_file_path):
        logging.info('data provider already exists in %s'%dp_file_path)
    else:
        data_provider = dp_class()
        data_provider.create(config)
        logging.info('build data provider and save into %s' % dp_file_path)

