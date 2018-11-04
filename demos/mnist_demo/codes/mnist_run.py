__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import importlib
import logging

from ml_idiot.solver.BasicSolver import BasicSolver as Solver

logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
    datefmt  = '%Y-%m-%d %A %H:%M:%S'
)


def main(config):
    solver = Solver(config)
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_set_name', type=str, default='MNIST')
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='CNN')
    parser.add_argument('-f', '--file', dest='config_file', type=str, default=None)
    args = parser.parse_args()
    config = vars(args)
    if config['config_file'] is None:
        config['config_file'] = '%s_%s_Config' % (config['data_set_name'], config['model_mane'])
    config.update(getattr(importlib.import_module('configures.%s' % config['config_file']), 'config'))
    main(config)
