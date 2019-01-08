__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import importlib
import logging
from solver.BasicSolver import BasicSolver as Solver
from demos.iris_demo.codes.settings import PROJECT_HOME


logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s:%(filename)s:%(levelname)s  %(message)s',
    datefmt  = '%Y-%m-%d-%H:%M:%S'
)


def main(config):
    solver = Solver(config)
    solver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', dest='data_set_name', type=str, default='iris')
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='LgbmGbdt', help="LR/SJLR/SKLearnLR/LgbmGbdt")
    parser.add_argument('-f', '--file', dest='config_file', type=str, default=None)
    args = parser.parse_args()
    config = vars(args)

    config["proj_folder"] = PROJECT_HOME
    main(config)
