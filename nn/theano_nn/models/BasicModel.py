__author__ = 'SongJun-Dell'
import logging
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class BasicModel(object):
    def __init__(self, state):
        self.state = state
        self.floatX = theano.config.floatX
        self.layers = list()
        self.params = list()
        self.regularize_params = list()

    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        np.savez(filename, **vals)

    def load(self, filename):
        """
        Load the model.
        """
        vals = np.load(filename)
        for p in self.params:
            if p.name in vals:
                logging.debug('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                if p.get_value().shape != vals[p.name].shape:
                    raise Exception('Shape mismatch: {} != {} for {}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                p.set_value(vals[p.name])
            else:
                logging.error('No parameter {} given: default initialization used'.format(p.name))
                unknown = set(vals.keys()) - {p.name for p in self.params}
                if len(unknown):
                    logging.error('Unknown parameters {} given'.format(unknown))

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        self.regularize_params += layer.regularize_params
        return layer

    def check(self):
        num_params = 0
        num_regulirize_params = 0

        for layer in self.layers:
            num_params += len(layer.params)
            num_regulirize_params += len(layer.regularize_params)

        assert( num_params==len(self.params), 'params number not equal!')
        # assert( num_grad_params==len(self.grad_params), 'grad params number not equal!')
        assert( num_regulirize_params==len(self.regularize_params), 'regularize params number not equal!')
    def get_regularization(self):
        reg_value = 0
        for p in self.regularize_params:
            reg_value += p.norm(L=2)
        return reg_value

    def create_updates(self, loss, params, method='rmsprop', updates=None, gradients=None):
        lr = theano.shared(np.float64(self.state['learning_rate']).astype(theano.config.floatX))
        grad_clip = theano.shared(np.float64(self.state['grad_clip']).astype(theano.config.floatX))
        ngrad_clip = theano.shared(np.float64(-self.state['grad_clip']).astype(theano.config.floatX))
        momentum = theano.shared(np.float64(self.state['momentum']).astype(theano.config.floatX))
        decay_rate = theano.shared(np.float64(self.state['decay_rate']).astype(theano.config.floatX))
        smooth_eps = theano.shared(np.float64(self.state['smooth_eps']).astype(theano.config.floatX))

        gcaches   = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in params]
        gparams = T.grad(loss, params) if gradients is None else gradients

        if updates is None:
            updates = OrderedDict()
        if method == 'rmsprop':
            for gparam, param, gcache in zip(gparams, params, gcaches):
                gparam = T.switch(T.ge(gparam, grad_clip), grad_clip, gparam)
                gparam = T.switch(T.le(gparam, ngrad_clip), ngrad_clip, gparam)

                updates[gcache] = gcache * decay_rate + (1.0 - decay_rate) * gparam ** 2
                gparam = gparam / T.sqrt(updates[gcache] + smooth_eps)
                updates[param] = param - gparam * lr
            return updates, gcaches, grad_clip, lr, ngrad_clip
