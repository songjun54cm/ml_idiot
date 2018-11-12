__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import abc
from ml_idiot.ml_models.IterationModels import IterationModel


class IterationFBModel(IterationModel):
    def __init__(self,config):
        super(IterationFBModel, self).__init__(config)

    @abc.abstractmethod
    def forward_batch(self, batch_data):
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

    def train_batch(self, batch_data, optimizer):
        forward_res = self.forward_batch(batch_data)
        batch_loss = forward_res["batch_loss"]
        score_loss = forward_res["score_loss"]
        regu_loss = forward_res["regu_loss"]
        grad_params = self.backward_batch(batch_data, forward_res)
        optimizer.optimize_model(self, grad_params)
        train_res = {
            "batch_loss": batch_loss,
            "score_loss": score_loss,
            "regu_loss": regu_loss,
        }
        return train_res