__author__ = 'JunSong<songjun54cm@gmail.com>'
from ml_idiot.trainer.IterationTrainer import IterationTrainer


class SJLRTrainer(IterationTrainer):
    def __init__(self, config):
        super(SJLRTrainer, self).__init__(config)
