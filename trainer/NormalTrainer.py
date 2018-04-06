__author__ = 'JunSong<songjun54cm@gmail.com>'
import json


from ml_idiot.trainer.BasicTrainer import BasicTrainer


class NormalTrainer(BasicTrainer):
    def __init__(self, config):
        super(NormalTrainer, self).__init__(config)

    def train(self, model, data_provider, tester):
        """
        train model using data provider
        :param model:
        :param data_provider:
        :param tester:
        :return:    None
        """
        self.create_out_folder()
        self.create_log_files()
        self.create_checkpoint_dir()
        self.log_train_message(json.dumps(self.config, indent=2), file_name='model_config.txt')

        self.train_model(model, data_provider, tester)

        self.log_valid_csv_message(self.top_performance_csv_message, 'top_performance_log.csv')
        self.log_train_message(self.top_performance_message)
        return self.last_save_model_file_path, self.top_performance_valid_res

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
