__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2019/1/7
import argparse
import lightgbm as lgb
from ml_models.NormalModel import NormalModel


class LgbmGbdtModel(NormalModel):
    def __init__(self):
        super(LgbmGbdtModel, self).__init__(config)
        self.num_round = 0
        self.gbdt = None

    def create(self, config):
        self.params["num_leaves"] = config.get("num_leaves", 31)
        self.params["num_trees"] = config.get("num_trees", 100)
        self.params["objective"] = config.get("objective", "binary")
        self.params["metric"] = config.get("metric", ["auc", "binary_logloss"])
        self.num_round = config.get("num_round", 10)

    def train(self, train_data):
        self.gbdt = lgb.train(self.params, train_data, self.num_round)

    def train_early_stop(self, train_data, valid_sets, early_stop_round=10):
        self.gbdt = lgb.train(self.params, train_data, self.num_round, valid_sets=valid_sets,
                              early_stopping_rounds=early_stop_round)
        # self.gbdt.save_model("model.txt", num_iteration=self.gbdt.best_iteration)

    def save(self, path):
        self.gbdt.save_model(path)

    def load(self, path):
        self.gbdt = lgb.Booster(model_file=path)

    def to_json(self):
        return self.gbdt.dump_model()

    def predict(self, in_data):
        return self.gbdt.predict(in_data)