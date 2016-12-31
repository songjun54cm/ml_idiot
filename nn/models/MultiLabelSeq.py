__author__ = 'SongJun-Dell'
import numpy as np
from ml_idiot.nn.models.SeqModel import SeqModel
from ml_idiot.utils.loss_functions import distance_loss, grad_distance_loss

class MultiLabelSeq(SeqModel):
    def __init__(self, state):
        super(MultiLabelSeq, self).__init__(state)
        self.loss_func = distance_loss
        self.grad_loss_func = grad_distance_loss

    def loss_one_sample(self, target_feas, pred_feas):
        loss = self.loss_func(pred_feas, target_feas)
        return loss

    def grad_loss_one_sample(self, pred_vecs, target_vecs):
        return self.grad_loss_func(pred_vecs, target_vecs)

    def predict_sample(self, sample_data, mode):
        pred_fea, cache = self.forward_sample(sample_data, mode)
        pred_fea[pred_fea>=0.5] = 1.0
        pred_fea[pred_fea<0.5] = 0.0
        if mode == 'test':
            return pred_fea
        elif mode in ['train', 'gc']:
            return pred_fea, cache
        else:
            raise StandardError('mode error.')

    # need to be implemented
    def forward_sample(self, sample_data, mode):
        raise NotImplementedError('forward sample not implemented.')

    def backward_sample(self, grad_params, forward_sample_cache):
        raise NotImplementedError('backward sample not implemented.')