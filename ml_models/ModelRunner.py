__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/1
import argparse
import importlib
import json
import logging
from ml_idiot.utils.utils import update_config


def create_dp(config, create=True):
    print('create data provider...')
    DataProvider = getattr(importlib.import_module('data_providers.%sDataProvider'%config['model_name']),
                           '%sDataProvider'%config['model_name'])
    data_provider = DataProvider()
    if create:
        data_provider.create(config)
        print('train num: %d, valid num: %d, test num: %d' %
              (data_provider.split_size('train'), data_provider.split_size('valid'), data_provider.split_size('test')))
    return data_provider


def create_model(state, dp):
    print('create model...')
    MLModel = getattr(importlib.import_module('models.%s' % state['model_name']),
                      '%s'%state['model_name'])
    model = MLModel(state, dp)
    return model


def create_solver(state):
    print('create solver...')
    MLSolver = getattr(importlib.import_module('solvers.%sSolver' % state['model_name']),
                       '%sSolver'%state['model_name'])
    solver = MLSolver(state)
    return solver


def init_config(user_config, dp):
    print('init config...')
    config = getattr(importlib.import_module('configs.%sConfig'%user_config['model_name']), user_config['data_set_name'])
    # if '__builtins__' in config: config.pop('__builtins__')
    config = update_config(config, user_config)
    return config


def main(user_config):
    data_provider = create_dp(user_config)
    config = init_config(user_config, data_provider)
    model = create_model(config, data_provider)
    solver = create_solver(config)
    if config['mode'] == 'train':
        logging.info('start training...')
        logging.info(json.dumps(config, indent=2))
        check_point_path, top_performance_res = solver.train(model, data_provider, 'normal')
        logging.info('training finish.')
        logging.info('model has been saved into %s'% check_point_path)
    elif config['mode'] == 'test':
        solver.test(model, data_provider, 'normal')
    else:
        raise Exception('run mode error')
    logging.info('finish running')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)

