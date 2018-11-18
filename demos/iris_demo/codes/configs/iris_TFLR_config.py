__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse


config = {
    "data_set_name": "iris",
    "model_name": "TFLRModel",
    "data_provider": "SJLRDataProvider",
    "trainer": "SJLRTrainer",
    "tester": "SJLRTester",
    "evaluator": "SJLREvaluator",
    "optimizer": "null",
    "max_epoch": 10,
    "batch_size": 10,
    "valid_batch_size": 10,
    'model_config': {
        "fea_size": 4,
        "optimizer": "sgd",
        "learning_rate": 0.01,
    },
}