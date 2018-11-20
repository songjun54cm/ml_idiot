__author__ = 'JunSong<songjun54cm@gmail.com>'
import abc
import tensorflow as tf
from ml_idiot.ml_models.IterationModels import IterationModel




class TensorFlowModel(IterationModel):
    def __init__(self, config):
        super(TensorFlowModel, self).__init__(config)
        self.session = None
        self.loss = None
        self.score_loss = None

    @abc.abstractmethod
    def create_model(self, model_config):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, data):
        raise NotImplementedError

    def get_optimizer(self,config):
        if(config["optimizer"] == "sgd"):
            return tf.train.GradientDescentOptimizer(learning_rate=config["learning_rate"])