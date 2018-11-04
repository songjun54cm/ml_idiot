__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.trainer.SGDTrainer import SGDTrainer


class LRTrainer(SGDTrainer):
    def __init__(self, config):
        super(LRTrainer, self).__init__(config)