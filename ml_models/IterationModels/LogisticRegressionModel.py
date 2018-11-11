__author__ = 'JunSong<songjun@corp.netease.com>'
# Date: 2018/11/5
import numpy as np
from ml_idiot.ml_models.IterationModels.IterationFBModel import IterationFBModel
from gradient_check.CheckerModelGradient import CheckerModelGradient

class LRGradientChecker(CheckerModelGradient):
    def __init__(self, model):
        x = np.random.rand(3,4)
        y = np.asarray([1,1,0]).reshape((3,1))
        gc_data = {"x":x,"y":y}
        super(LRGradientChecker, self).__init__(model, gc_data)

class MainTestLRGradient():
    def __init__(self):
        pass

    def run(self):
        config = {}
        lr = LogisticRegressionModel(config)
        lr.check_gradient()

class LogisticRegressionModel(IterationFBModel):
    def __init__(self, config):
        super(LogisticRegressionModel, self).__init__(config)
        self.w = None
        self.w_name = None
        self.b = None
        self.b_name = None

    def create(self, config):
        self.w, self.w_name = self.add_params((config["fea_size"], 1), "w")
        self.w.fill(1.0)
        self.b, self.b_name = self.add_params((1,), "b")
        self.regularize_param_names.append(self.w_name)

    def forward_batch(self, batch_data):
        x = batch_data["x"] # x: n*d matrix
        y = batch_data["y"] # y: n*1 array
        s = np.dot(x, self.w) + self.b # n*d dot d*1 = n*1
        e = np.log(np.exp(s) + 1)
        q = y * s # n*1
        loss = np.sum(e-q, axis=0, keepdims=False)
        forward_res = {
            "batch_loss": loss,
            "score_loss": loss,
            "regu_loss": 0.0,
            "pred_feas" : s,
            "gth_vals": y
        }
        return forward_res

    def backward_batch(self, batch_data, forward_res):
        x = batch_data["x"]  # x: n*d matrix
        y = batch_data["y"]  # y: n*1 array
        pred_feas = forward_res["pred_feas"]
        gradients = {}
        e = np.exp(pred_feas)
        gb = (e / (1 + e)) - y
        gradients[self.w_name] = np.dot(np.transpose(x), gb)
        gradients[self.b_name] = np.sum(gb, axis=0,keepdims=False)
        return gradients

    def predict_batch(self, batch_data):
        f_res = self.forward_batch(batch_data)
        pred_vals = f_res["pred_feas"] > 0.5
        res = {
            "loss": f_res["batch_loss"],
            "pred_vals": pred_vals,
            "gth_vals": f_res["gth_vals"]
        }
        return res

    def get_loss(self, batch_data, mode=None):
        forward_res = self.forward_batch(batch_data)
        if(mode=="gc"):
            grds = self.backward_batch(batch_data, forward_res)
            return forward_res["batch_loss"], grds
        else:
            return forward_res["batch_loss"]

    def check_gradient(self):
        config = {
            "fea_size": 4
        }
        self.create(config)
        gChecker = LRGradientChecker(self)
        gChecker.check_gradient()
