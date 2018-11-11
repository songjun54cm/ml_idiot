__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.ml_models.IterationModels.LogisticRegressionModel import LogisticRegressionModel


class SJLRModel(LogisticRegressionModel):
    def __init__(self, config):
        super(SJLRModel, self).__init__(config)