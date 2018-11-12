__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
from ml_idiot.ml_models.IterationModels import IterationModel
from sklearn import linear_model


class SKLearnLRModel(IterationModel):
    def __init__(self,config):
        super(SKLearnLRModel, self).__init__(config)
        self.config = config
        self.lr = linear_model.SGDClassifier()
        self.canPreValid = False

    def create(self, config):
        self.lr = linear_model.SGDClassifier(loss="log",
                                             warm_start=True,
                                             learning_rate=config["learning_rate"],
                                             eta0=config["eta0"])

    def train_batch(self, batch_data, optimizer=None):
        x = batch_data["x"]
        y = batch_data["y"]
        self.lr.warm_start = True
        self.lr.max_iter = self.config["batch_iter_n"]
        self.lr.fit(x, y)
        train_res = {
            "batch_loss": 0.0,
            "score_loss": 0.0,
            "regu_loss": 0.0
        }
        return train_res

    def predict_batch(self, batch_data):
        x = batch_data["x"]
        y = batch_data["y"]
        pred_val = self.lr.predict(x)
        res = {
            "loss": 0.0,
            "pred_vals": pred_val,
            "gth_vals": y
        }
        return res

    def summary(self):
        print("coef: " + str(self.lr.coef_))
        print("intercept: %f" % self.lr.intercept_)
