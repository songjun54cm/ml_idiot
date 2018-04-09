__author__ = 'JunSong<songjun54cm@gmail.com>'
import importlib
import json
import logging

from ml_idiot.config import complete_config


def form_name(val, suffix):
    if val[-len(suffix):] == suffix:
        return val
    else:
        return '%s%s' % (val, suffix)


def create_data_provider(config, create=True):
    logging.info('create data provider: %s ...' % config['data_provider'])
    dp_cls_name = form_name(config['data_provider'], 'DataProvider')
    dp_cls = getattr(importlib.import_module('data_providers.%s'%dp_cls_name), dp_cls_name)
    data_provider = dp_cls()
    if create:
        data_provider.create(config)
    return data_provider


def create_model(config):
    logging.info('create model: %s ...' % config['model_name'])
    model_cls_name = form_name(config['model_name'], 'Model')
    model_cls = getattr(importlib.import_module('models.%s' % model_cls_name), model_cls_name)
    model = model_cls()
    model.create(config['model_config'])
    return model


def create_trainer(config):
    logging.info('create trainer: %s ...' % config['trainer'])
    trainer_cls_name = form_name(config['trainer'], 'Trainer')
    trainer_cls = getattr(importlib.import_module('ml_idiot.trainer.%s' % trainer_cls_name), trainer_cls_name)
    trainer = trainer_cls(config)
    return trainer


def create_tester(config):
    logging.info('create tester: %s ...' % config['tester'])
    tester_cls_name = form_name(config['tester'], 'Tester')
    tester_cls = getattr(importlib.import_module('ml_idiot.tester.%s' % tester_cls_name), tester_cls_name)
    tester = tester_cls(config)
    return tester


class BasicSolver(object):
    config = {}
    model = None
    data_provider = None
    trainer = None
    tester = None

    def __init__(self, config):
        super(BasicSolver, self).__init__()
        self.config = complete_config(config)
        self.data_provider = create_data_provider(config)
        self.model = create_model(config)
        self.trainer = create_trainer(config)
        self.tester = create_tester(config)

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

