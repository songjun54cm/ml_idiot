__author__ = 'SongJun-Dell'
import numpy as np
from ml_idiot.nn.models.BasicModel import BasicModel


class SeqModel(BasicModel):
    def __init__(self, state):
        super(SeqModel, self).__init__(state)

    @staticmethod
    def fullfillstate(state):
        return state

    def get_batch_loss(self, batch_data, mode='train'):
        batch_losses = list()
        # gc--gradient check...
        grad_params = self.init_grads() if mode in ['train','gc'] else None
        for data in batch_data:
            pred_feas, forward_sample_cache = self.forward_sample(data, mode)
            sample_loss = self.loss_one_sample(data['target_feas'], pred_feas)
            batch_losses.append(sample_loss)
            if mode in ['train', 'gc']:
                self.backward_sample(grad_params, forward_sample_cache)
            elif mode in ['test']:
                pass
            else:
                raise StandardError('mode error.')
        batch_loss = np.mean(batch_losses)
        if mode in ['train', 'gc']:
            for p in grad_params.keys():
                grad_params[p] /= len(batch_data)
        elif mode == 'test':
            pass
        else:
            raise StandardError('mode error')
        return batch_loss, grad_params

    def predict_batch(self, batch_data, mode='test'):
        preds = list()
        if mode == 'test':
            for data in batch_data:
                pred_sample = self.predict_sample(data, mode)
                preds.append(pred_sample)
            return preds
        elif mode in ['train', 'gc']:
            caches = list()
            for data in batch_data:
                pred_sample, cache = self.predict_sample(data, mode)
                preds.append(pred_sample)
                caches.append(cache)
            return preds, caches
        else:
            raise StandardError('mode error.')

    ###

    def forward_sample(self, sample_data, mode):
        raise NotImplementedError('forward sample not implemented')

    def loss_one_sample(self, target_fea, pred_fea):
        raise NotImplementedError('loss one sample not implemented')

    def backward_sample(self, grad_params, forward_sample_cache):
        raise NotImplementedError('backward sample not implemented')

    def predict_sample(self, sample_data, mode):
        raise NotImplementedError('predict sample not implemented')