__author__ = 'JunSong<songjun54cm@gmail.com>'
import abc
from ml_idiot.ml_models.BasicModel import BasicModel


class IterationModel(BasicModel):
    def __init__(self, config):
        super(IterationModel, self).__init__(config)

    @abc.abstractmethod
    def create(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def train_batch(self, batch_data, optimizer):
        raise NotImplementedError