__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.ml_models.IterationModels.IterationFBModel import IterationFBModel


class NNModel(IterationFBModel):
    def __init__(self, config):
        super(NNModel, self).__init__(config)
