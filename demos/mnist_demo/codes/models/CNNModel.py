__author__ = 'JunSong<songjun54cm@gmail.com>'
from ml_idiot.ml_models.BasicModel import BasicModel



class CNNModel(BasicModel):
    def __init__(self, config):
        super(CNNModel, self).__init__(config)

    def init_model(self):
        pass

    def forward_sample(self, sample_data):
        pass

    def backward_sample(self, forward_cache):
        pass

