__author__ = 'JunSong<songjun54cm@gmail.com>'
# Date: 2018/11/2
import argparse
import abc
from ml_idiot.ml_models.BasicModel import BasicModel


class NormalModel(BasicModel):
    """
    model trained by fit train data once
    """
    def __init__(self, config):
        super(NormalModel, self).__init__(config)

    @abc.abstractmethod
    def create(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, train_data):
        raise NotImplementedError

