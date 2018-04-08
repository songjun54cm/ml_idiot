__author__ = 'JunSong<songjun54cm@gmail.com>'
import tensorflow as tf



class TensorFlowModel(object):
    def __init__(self):
        self.session = None
        self.loss = None
        self.score_loss = None

    def create_model(self, model_config):
        pass

    def train(self, data):
        if self.session is None:
            self.session = tf.Session()
            init =