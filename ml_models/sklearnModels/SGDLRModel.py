__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.ml_models.IterationModels import IterationModel
from sklearn import linear_model
import numpy as np

class SGDLRModel(IterationModel):
    def __init__(self, config):
        super(SGDLRModel, self).__init__(config)

    def create(self, config):
        self.lr = linear_model.SGDClassifier(loss="log",
                                             warm_start=True,
                                             learning_rate=config["learning_rate"],
                                             eta0=config["eta0"])

    def train_batch(self, batch_data, optimizer=None):
        x = batch_data["x"]
        y = batch_data["y"]
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int)
        self.lr.warm_start = True
        self.lr.max_iter = self.config["iter_n"]
        self.lr.fit(x, y)
        self.summary()

    def summary(self):
        print("coef: " + str(self.lr.coef_))
        print("intercept: %f" % self.lr.intercept_)