__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
import abc
from ml_idiot.ml_models.BasicModel import BasicModel


class NormalModel(BasicModel):
    def __init__(self, config):
        super(NormalModel, self).__init__(config)

    @abc.abstractmethod
    def create(self, config):
        raise NotImplementedError

    @abc.abstractmethod
    def forward_batch_loss(self, batch_data):
        """
        forward one batch data
        :param batch_samples:   list of samples
        :return: forward_res = {
            batch_loss:
            score_loss:
            regu_loss:
            ...
        }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def backward_batch(self, batch_data, forward_res):
        """
        train one batch data
        :param loss:    loss
        :param batch_samples:   list of samples
        :param forward_res: {}
        :return:    gradient = {param_name:gradient_value}
        """
        raise NotImplementedError

    def train_batch(self, batch_data):
        forward_res = self.forward_batch_loss(batch_data)
        batch_loss = forward_res["batch_loss"]
        score_loss = forward_res["score_loss"]
        regu_loss = forward_res["regu_loss"]
        grad_params = self.backward_batch(batch_loss, batch_data, forward_res)
        return batch_loss, score_loss, regu_loss, grad_params
