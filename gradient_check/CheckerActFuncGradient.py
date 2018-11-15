__author__ = 'SongJun-Dell'
from ml_idiot.gradient_check.CheckerGradient import CheckerGraident


class CheckerActFuncGradient(CheckerGraident):
    def __init__(self, act_func, grad_func, input_data, gth_out):
        super(CheckerActFuncGradient, self).__init__()
        self.act_func = act_func
        self.grad_func = grad_func
        self.input_data = input_data
        self.gth_out = gth_out
