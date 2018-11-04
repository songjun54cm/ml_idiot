__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.ml_models.NormalModel import NormalModel
from sklearn import linear_model
import numpy as np


class LRModel(NormalModel):
    def __init__(self, config, dp=None):
        super(LRModel, self).__init__(config)
        self.lr = linear_model.SGDClassifier()

    def create(self,config):
        self.lr = linear_model.SGDClassifier(loss="log",
                                             warm_start=True,
                                             learning_rate=config["learning_rate"],
                                             eta0=config["eta0"])

    def loss_batch_predict(self, batch_samples):
        x = np.asarray([s[0] for s in batch_samples])
        gth_y = [s[1] for s in batch_samples]
        prd_y = self.lr.predict(x)
        return -1, list(prd_y), gth_y

    def train_one_time(self, dp, iter_n=10):
        x = []
        y = []
        for (fea, label) in dp.get_split("train"):
            x.append(fea)
            y.append(label)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int)
        self.lr.warm_start = True
        self.lr.max_iter = iter_n
        self.lr.fit(x, y)

    def summary(self):
        print("coef: " + str(self.lr.coef_))
        print("intercept: %f" % self.lr.intercept_)