__author__ = 'JunSong<songjun54cm@gmail.com>'
import importlib
import json
import logging
from ml_idiot.utils.utils import update_config



def form_name(val, suffix):
    if val[-len(suffix):] == suffix:
        return val
    else:
        return '%s%s' % (val, suffix)


def create_data_provider(config, create=True):
    logging.info('create data provider: %s ...' % config['data_provider'])
    dp_cls_name = form_name(config['data_provider'], 'DataProvider')
    dp_cls = getattr(importlib.import_module('data_providers.%s' % dp_cls_name), dp_cls_name)
    data_provider = dp_cls()
    if create:
        data_provider.create(config)
    return data_provider


def create_model(config):
    logging.info('create model: %s ...' % config['model_name'])
    model_cls_name = form_name(config['model_name'], 'Model')
    model_cls = getattr(importlib.import_module('models.%s' % model_cls_name), model_cls_name)
    model = model_cls(config)
    model.create(config['model_config'])
    return model


def create_trainer(config):
    logging.info('create trainer: %s ...' % config['trainer'])
    trainer_cls_name = form_name(config['trainer'], 'Trainer')
    trainer_cls = getattr(importlib.import_module('trainers.%s' % trainer_cls_name), trainer_cls_name)
    trainer = trainer_cls(config)
    return trainer


def create_tester(config):
    logging.info('create tester: %s ...' % config['tester'])
    tester_cls_name = form_name(config['tester'], 'Tester')
    tester_cls = getattr(importlib.import_module('testers.%s' % tester_cls_name), tester_cls_name)
    tester = tester_cls(config)
    return tester


def init_config(user_config):
    print('init config...')
    if user_config['config_file'] is None:
        user_config['config_file'] = '%s_%s_config' % (user_config['data_set_name'], user_config['model_name'])
    user_config.update(getattr(importlib.import_module('configs.%s' % user_config['config_file']), 'config'))
    user_config["data_provider"] = user_config.get("data_provider", "%sDataProvider" % user_config["model_name"])
    user_config["trainer"] = user_config.get("trainer", "%sTrainer" % user_config["model_name"])
    user_config["tester"] = user_config.get("tester", "%sTester" % user_config["model_name"])
    user_config["evaluator"] = user_config.get("evaluator", "%sEvaluator" % user_config["model_name"])
    user_config["mode"] = user_config.get("mode", "train")
    return user_config


class BasicSolver(object):
    config = {}
    model = None
    data_provider = None
    trainer = None
    tester = None

    def __init__(self, config):
        super(BasicSolver, self).__init__()
        self.config = init_config(config)
        self.data_provider = create_data_provider(self.config)
        self.model = create_model(self.config)
        self.trainer = create_trainer(self.config)
        self.tester = create_tester(self.config)
        self.prepare_solver()

    def prepare_solver(self):
        self.trainer.prepare_trainer(self)

    def run(self):
        if self.config['mode'] in ['train', 'debug']:
            logging.info('start training ...')
            logging.info(json.dumps(self.config, indent=2))
            check_point_path, top_performance_res = self.trainer.train(self.model, self.data_provider, self.tester)
            logging.info('training finish.')
            logging.info('model has been saved into %s' % check_point_path)
        elif self.config['model'] == 'test':
            self.tester.test(self.model, self.data_provider)
        else:
            raise BaseException('mode error')
        logging.info('finish running.')

