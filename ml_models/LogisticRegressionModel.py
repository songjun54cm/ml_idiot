__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/5
import numpy as np
from ml_idiot.ml_models.NormalModel import NormalModel


class LogisticRegressionModel(NormalModel):
    def __init__(self, config):
        super(LogisticRegressionModel, self).__init__(config)

