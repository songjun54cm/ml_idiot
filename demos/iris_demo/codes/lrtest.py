__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_models.IterationModels.LogisticRegressionModel import LogisticRegressionModel


def main(config):
    lr = LogisticRegressionModel(config)
    lr.check_gradient()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)