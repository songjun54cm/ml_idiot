__author__ = 'JunSong'
import random
import numpy as np
import cPickle as pickle
from ml_idiot.utils.utils import get_data_splits
def create_model(model_class, model_file_path):
    print 'load model from %s' % model_file_path
    model_dump = pickle.load(open(model_file_path,'rb'))
    state = model_dump['state']
    model = model_class(state)
    model.load_from_dump(model_dump)
    return model

def get_model_batch_test_result(model, batch_data):
    res = model.test_on_batch(batch_data)
    return res

def get_train_loss_one_batch(model, batch_data, mode):
    loss, grad_params = model.get_batch_loss(batch_data, mode=mode)
    return {'loss': loss, 'grad_params': grad_params}

def get_batch_predict_candidates(model, batch_data):
    res = model.get_predict_candidates(batch_data)
    return res

class BasicModel(object):
    def __init__(self, state):
        # Parameters of the model
        self.state = state
        self.params = dict()
        self.grad_params = dict()
        self.grad_cache = dict()
        self.regularize_param_names = list()
        self.layers = list()

    def print_params(self):
        for p,v in self.params.items():
            print 'parameter: %s, shape: %s' % (p, str(v.shape))
            print 'values:'
            print v

    def save(self, filename):
        """
        Save the models to file `filename`
        """
        model_dump = {
            'params': self.params,
            'state': self.state
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_dump, f)

    def load(self, filename):
        """
        Load the models.
        """
        model_dump = pickle.load(open(filename, 'rb'))
        self.load_from_dump(model_dump)

    def load_from_dump(self, model_dump):
        model_params = model_dump['params']
        for p in self.params.keys():
            self.params[p].setfield(model_params[p], dtype=self.params[p].dtype)
        self.state = model_dump['state']

    def load_params(self, model_params):
        for p in self.params.keys():
            self.params[p].setfield(model_params[p], dtype=self.params[p].dtype)

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params.update(layer.params)
        self.regularize_param_names += layer.regularize_param_names
        return layer

    def get_regularization(self):
        reg_value = 0
        for p in self.regularize_param_names:
            reg_value += np.linalg.norm(self.params[p])**2
        reg_value /= 2
        return reg_value

    def grad_regularization(self, grad_params):
        for p in self.regularize_param_names:
            grad_params[p] += self.state['regularize_rate'] * self.params[p]

    def init_grads(self):
        grad_params = dict()
        for layer in self.layers:
            lay_grads = layer.init_grads()
            grad_params.update(lay_grads)
        return grad_params

    def clear_grads(self, grad_params):
        for p in grad_params:
            grad_params[p].fill(0)

    def get_params(self):
        return self.params

    def params_scale(self, val):
        for p in self.params.keys():
            self.params[p] *= val

    def check(self):
        num_params = 0
        num_regulirize_params = 0

        for layer in self.layers:
            num_params += len(layer.params)
            num_regulirize_params += len(layer.regularize_param_names)

        assert( num_params==len(self.params), 'params number not equal!')
        # assert( num_grad_params==len(self.grad_params), 'grad params number not equal!')
        assert( num_regulirize_params==len(self.regularize_param_names), 'regularize params number not equal!')

    def merge_grads(self, grad_params, grads, scale=1.0):
        if grad_params is None:
            grad_params = dict()
            for p in grads.keys():
                grad_params[p] = grads[p]/scale
        else:
            for p in grads.keys():
                grad_params[p] += (grads[p]/scale)

        return grad_params

    def train_one_batch(self, batch_data, pool=None, num_processes=0, mode='train'):
        if pool is None:
            train_res = get_train_loss_one_batch(self, batch_data, mode)
            train_loss = train_res['loss']
            grad_params = train_res['grad_params']
        else:
            batch_splits = get_data_splits(num_processes, batch_data)
            train_loss = 0
            grad_params = None
            pool_results = list()
            for pro in xrange(num_processes):
                data_samples = batch_splits[pro]
                tmp_result = pool.apply_async(get_train_loss_one_batch, (self, data_samples, mode))
                pool_results.append(tmp_result)

            for p in xrange(num_processes):
                split_res = pool_results[p].get()
                train_loss += split_res['loss']
                if mode=='train':
                    grad_params = self.merge_grads(grad_params, split_res['grad_params'], scale=float(num_processes))

            train_loss /= num_processes

        if mode=='train':
            self.grad_regularization(grad_params)
        # print 'train_cost %f' % train_cost
        train_loss = train_loss + self.state['regularize_rate'] * self.get_regularization()
        return train_loss, grad_params

    def get_loss(self, batch_data, pool=None, num_process=0, mode='test'):
        loss, grad_params = self.train_one_batch(batch_data, pool, num_process, mode)
        if mode == 'train':
            return loss, grad_params
        elif mode == 'test':
            return loss
        else:
            raise StandardError('mode error')

    # functions need to be implemented.
    def get_batch_loss(self, batch_data, mode='train'):
        """

        :param batch_data:
        :param mode:
        :return:
        mode='train': return batch_loss, grad_params
        mode='test': return batch_loss, None
        """
        raise(NotImplementedError('get_batch_loss not implemented.'))