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
    def train(self, train_data, valid_data=None, test_data=None):
        raise NotImplementedError

    @abc.abstractmethod
    def predict_batch(self, batch_data):
        """
        predict result of one batch data
        :param batch_data:
        :return: res = {
            "loss": batch_loss,
            "pred_vals": pred_vals,
            "gth_vals": gth_vals
        }
        """
        raise NotImplementedError


