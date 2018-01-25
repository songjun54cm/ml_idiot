__author__ = 'SongJun-Dell'
from ml_idiot.utils.utils import path_join

def get_dp_file_path(state):
    return path_join(state['data_folder'], '%s_%s_data_provider.pkl'%(state['model_name'], state['data_set_name']))

def pre_build_dp(config, dp_class):
    from ml_idiot.utils.utils import get_data_folder
    from ml_idiot.data_provider import get_dp_file_path
    config['data_folder'] = get_data_folder(config)
    import os
    dp_file_path = get_dp_file_path(config)
    if os.path.exists(dp_file_path):
        print 'data provider already exists in %s'%dp_file_path
    else:
        data_provider = dp_class()
        data_provider.build(config)
        data_provider.save(dp_file_path)
        print('build data provider and save into %s' % dp_file_path)