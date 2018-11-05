__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
from ml_idiot.ml_models.BasicModel import BasicModel


class NormalModel(BasicModel):
    def __init__(self, config):
        super(NormalModel, self).__init__(config)

    def create(self,config):
        print("do nothing, pass.")
        pass
