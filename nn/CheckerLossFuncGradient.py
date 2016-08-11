__author__ = 'SongJun-Dell'
from CheckerGradient import CheckerGraident
class CheckerLossFuncGradient(CheckerGraident):
    def __init__(self, act_func, grad_func, input_data):
        super(CheckerLossFuncGradient, self).__init__()
        self.loss_func = act_func
        self.grad_func = grad_func
        self.input_data = input_data

    def get_loss(self, mode):
        loss, train_cache = self.loss_func(self.input_data, mode=mode)
        if mode=='train':
            grad_params = self.grad_func(train_cache)
            return loss, grad_params
        elif mode=='test':
            return loss
        else:
            raise StandardError('mode error.')

    def get_gc_params(self):
        return self.input_data