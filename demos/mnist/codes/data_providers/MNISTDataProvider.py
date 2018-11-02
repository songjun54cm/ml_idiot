__author__ = 'JunSong<songjun54cm@gmail.com>'
from settings import DATA_HOME
import os
import logging
from ml_idiot.utils.save_load import data_load


class MNISTDataProvider(object):
    def __init__(self):
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None

    def create(self):
        raw_data_path = os.path.join(DATA_HOME, 'mnist', 'mnist_data')
        datas = data_load(raw_data_path)
        self.train_images = datas['train_images']
        self.train_labels = datas['train_labels']
        self.test_images = datas['test_images']
        self.test_labels = datas['test_labels']

        self.summary()

    def summary(self):
        logging.info("train images shape: %s" % str(self.train_images.shape))
        logging.info("train labels shape: %s" % str(self.train_labels.shape))
        logging.info("test images shape: %s" % str(self.test_images.shape))
        logging.info("test labels shape: %s" % str(self.test_labels.shape))
