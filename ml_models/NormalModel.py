__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
from ml_idiot.ml_models.BasicModel import BasicModel


class NormalModel(BasicModel):
    def __init__(self, config):
        super(NormalModel, self).__init__(config)


    def create(self, config):
        print("do nothing, pass.")
        pass

    def train_batch(self, batch_data):
        loss = self.forward_batch_loss(batch_data)
        grad_params = self.backward_batch(loss, batch_data)
        self.optimizer.optimize_model(self, grad_params)
        return loss

    def forward_batch_loss(self, batch_data):
        """
        forward one batch data
        :param batch_samples:   list of samples
        :return:    loss, pred_vals, gth_vals
        """
        raise NotImplementedError

    def backward_batch(self, loss, batch_data):
        """
        train one batch data
        :param loss:    loss
        :param batch_samples:   list of samples
        :return:    gradient
        """
        raise NotImplementedError