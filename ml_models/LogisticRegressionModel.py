__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/5
import numpy as np
from ml_idiot.ml_models.NormalModel import NormalModel


class LogisticRegressionModel(NormalModel):
    def __init__(self, config):
        super(LogisticRegressionModel, self).__init__(config)
        self.w = None
        self.w_name = None
        self.b = None
        self.b_name = None

    def create(self, config):
        self.w, self.w_name = self.add_params((config["fea_size"], 1), "w")
        self.b, self.b_name = self.add_params((1,), "b")
        self.regularize_param_names.append(self.w_name)

    def forward_batch_loss(self, batch_data):
        x = batch_data["x"]
        y = batch_data["y"]


    def backward_batch(self, batch_loss, batch_data):
