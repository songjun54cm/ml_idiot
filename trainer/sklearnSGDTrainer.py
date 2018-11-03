__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/2
import argparse
from ml_idiot.trainer.NormalTrainer import NormalTrainer


class sklearnSGDTrainer(NormalTrainer):
    def __init__(self, config):
        super(sklearnSGDTrainer, self).__init__(config)
        self.config = config

    def train_model(self, model, data_provider, tester):
        iter_n = 0
        while iter_n < self.config["max_iter"]:
            model.train_one_time(data_provider, iter_n=self.config["test_interval"])
            tester.test(model, data_provider)
            iter_n += self.config["test_interval"]