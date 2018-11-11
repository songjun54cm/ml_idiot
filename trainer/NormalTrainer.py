__author__ = 'JunSong<songjun54cm@gmail.com>'
import json


from ml_idiot.trainer.BasicTrainer import BasicTrainer


class NormalTrainer(BasicTrainer):
    """
    train model by calling .train once
    """
    def __init__(self, config):
        super(NormalTrainer, self).__init__(config)

    def train_model(self, model, data_provider, tester):
        """
        train model using data provider
        :param model:   model to be trained
        :param data_provider:   provide data
        :return:    None
        """
        model.train(data_provider.get_split('train'))
        res = tester.test(model, data_provider)
        self.top_performance_csv_message = self.form_valid_csv_message(res)
        self.top_performance_message = self.form_valid_message(res)
