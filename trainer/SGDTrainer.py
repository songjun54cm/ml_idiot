__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.trainer.NormalTrainer import NormalTrainer


class SGDTrainer(NormalTrainer):
    def __init__(self, config):
        super(SGDTrainer, self).__init__(config)

    def train_model(self, model, data_provider, tester):
        iter_n = 0
        while iter_n < self.config["max_iter"]:
            model.train_one_time(data_provider, iter_n=self.config["test_interval"])
            res = tester.test(model, data_provider)
            message = self.form_valid_message(res)
            self.log_train_message(message)
            csv_message = self.form_valid_csv_message(res)
            self.log_train_csv(csv_message)
            iter_n += self.config["test_interval"]
            self.save_or_not(res["valid"], model, csv_message, message)
        model.summary()