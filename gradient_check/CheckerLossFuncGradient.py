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

def main(func_name):
    import numpy as np
    if func_name == 'distance_func':
        from ml_idiot.utils.loss_functions import distance_loss, grad_distance_loss
        target_fea = np.random.random((5,3))
        gc_data = {'pred_feas': np.random.random((5,3))}
        def loss_func(data, mode=None):
            loss = distance_loss(data['pred_feas'], target_fea)
            train_cache = data
            return loss, train_cache
        def grad_func(cache):
            grad = grad_distance_loss(cache['pred_feas'], target_fea)
            return {'pred_feas': grad}

    if func_name == 'tanh':
        from ml_idiot.nn.layers.BasicLayer import tanh_func, grad_tanh
        gc_data = {'in_vecs': np.random.random((5,3))}
        def loss_func(data, mode=None):
            loss = np.sum(tanh_func(data['in_vecs']))
            train_cache = data
            return loss, train_cache
        def grad_func(cache):
            grad = grad_tanh(tanh_func(cache['in_vecs']))
            return {'in_vecs': grad}

    gck = CheckerLossFuncGradient(loss_func, grad_func, gc_data)
    gck.check_gradient()

if __name__=='__main__':
    # func_name = 'distance_func'
    func_name = 'tanh'
    main(func_name)