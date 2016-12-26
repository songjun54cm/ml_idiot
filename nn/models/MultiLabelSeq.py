__author__ = 'SongJun-Dell'
import numpy as np
from ml_idiot.nn.models.BasicModel import BasicModel

class MultiLabelSeq(BasicModel):
    def __init__(self, state):
        super(MultiLabelSeq, self).__init__(state)

    def loss_one_sample(self, target_feas, pred_feas):
        loss = self.loss_func(pred_feas, target_feas)
        return loss

    def grad_loss_one_sample(self, pred_vecs, target_vecs):
        return self.grad_loss_func(pred_vecs, target_vecs)

    def get_batch_loss(self, batch_data, mode='train'):
        batch_losses = list()
        # gc--gradient check...
        grad_params = self.init_grads() if mode in ['train','gc'] else None
        for data in batch_data:
            pred_feas, forward_sample_cache = self.forward_sample(data, mode)
            sample_loss = self.loss_one_sample(data['target_feas'], pred_feas)
            batch_losses.append(sample_loss)
            if mode == 'train':
                self.backward_sample(grad_params, forward_sample_cache)
        batch_loss = np.mean(batch_losses)
        if mode == 'train':
            for p in grad_params.keys():
                grad_params[p] /= len(batch_data)
        return batch_loss, grad_params

    def predict_batch(self, batch_data, mode):
        preds = list()
        if mode == 'test':
            for data in batch_data:
                pred_sample = self.predict_sample(data, mode)
                preds.append(pred_sample)
            return preds
        elif mode == 'train':
            caches = list()
            for data in batch_data:
                pred_sample, cache = self.predict_sample(data, mode)
                preds.append(pred_sample)
                caches.append(cache)
            return preds, caches
        else:
            raise StandardError('mode error.')

    def predict_sample(self, sample_data, mode):
        pred_fea, cache = self.forward_sample(sample_data, mode)
        pred_fea[pred_fea>=0.5] = 1.0
        pred_fea[pred_fea<0.5] = 0.0
        if mode == 'test':
            return pred_fea
        else:
            return pred_fea, cache

    # need to be implemented
    def forward_sample(self, sample_data, mode):
        raise NotImplementedError('forward sample not implemented.')

    def backward_sample(self, grad_params, forward_sample_cache):
        raise NotImplementedError('backward sample not implemented.')