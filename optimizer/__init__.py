__author__ = 'JunSong<songjun54cm@gmail.com>'
import numpy as np

class BasicOptimizer(object):
    def __init__(self, config):
        self.grad_clip = config.get('grad_clip', 0.0)
        self.momentum = config.get('momentum', 0.0)
        self.smooth_eps = config.get('smooth_eps', 1e-8)
        self.learning_rate = config['learning_rate']
        self.learning_rate_decay = config.get('learning_rate_decay', 1.0)
        self.grad_cache = dict()

    def decay_learning_rate(self):
        self.learning_rate *= self.learning_rate_decay

    def optimize_model(self, model, grad_params):
        raise NotImplementedError('optimize_model not implemented.')

class SGD(BasicOptimizer):
    def __init__(self, config):
        super(SGD, self).__init__(config)

    def optimize_model(self, model, grad_params):
        for p in grad_params.keys():
            grad_p = grad_params[p]
            if self.grad_clip > 0:
                grad_p = np.minimum(grad_p, self.grad_clip)
                grad_p = np.maximum(grad_p, -self.grad_clip)
            grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
            grad_p_cache = self.momentum * grad_p_cache + self.learning_rate * grad_p
            dx_p = -grad_p_cache

            self.grad_cache[p] = grad_p_cache
            model.params[p] += dx_p

class RMSProp(BasicOptimizer):
    def __init__(self, config):
        super(RMSProp, self).__init__(config)
        self.rho = config.get('rho', 0.95)

    def optimize_model(self, model, grad_params):
        for p in grad_params.keys():
            grad_p = grad_params[p]
            # clip gradient
            if self.grad_clip > 0:
                grad_p = np.minimum(grad_p, self.grad_clip)
                grad_p = np.maximum(grad_p, -self.grad_clip)
            grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
            grad_p_cache = grad_p_cache * self.rho + (1.0-self.rho) * grad_p ** 2
            dx_p = -(self.learning_rate*grad_p)/np.sqrt(grad_p_cache + self.smooth_eps)

            self.grad_cache[p] = grad_p_cache
            model.params[p] += dx_p

class Adadelta(BasicOptimizer):
    def __init__(self, config):
        super(Adadelta, self).__init__(config)
        self.rho = config.get('rho', 0.95)
        self.delta_cache = dict()

    def optimize_model(self, model, grad_params):
        for p in grad_params.keys():
            grad_p = grad_params[p]
            # clip gradient
            if self.grad_clip > 0:
                grad_p = np.minimum(grad_p, self.grad_clip)
                grad_p = np.maximum(grad_p, -self.grad_clip)
            grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
            grad_p_cache = self.rho * grad_p_cache + (1.0-self.rho)*grad_p**2
            delta_p_cache = self.delta_cache.get(p, np.zeros(grad_p.shape))
            dx_p = grad_p * np.sqrt(delta_p_cache + self.smooth_eps) / np.sqrt(grad_p_cache + self.smooth_eps)

            self.grad_cache[p] = grad_p_cache
            model.params[p] -= dx_p
            delta_p_cache = self.rho * delta_p_cache + (1.0-self.rho)*dx_p**2
            self.delta_cache[p] = delta_p_cache

class Adagrad(BasicOptimizer):
    def __init__(self, config):
        super(Adagrad, self).__init__(config)

    def optimize_model(self, model, grad_params):
        for p in grad_params.keys():
            grad_p = grad_params[p]
            # clip gradient
            if self.grad_clip > 0:
                grad_p = np.minimum(grad_p, self.grad_clip)
                grad_p = np.maximum(grad_p, -self.grad_clip)
            grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
            grad_p_cache += grad_p**2
            dx_p = -self.learning_rate * grad_p / np.sqrt(grad_p_cache + self.smooth_eps)
            self.grad_cache[p] = grad_p_cache
            model.params[p] += dx_p

class Adam(BasicOptimizer):
    def __init__(self, config):
        super(Adam, self).__init__(config)
        self.beta1 = config.get('beta1', 0.9)
        self.beta2 = config.get('beta2', 0.999)
        self.t = 1
        self.grad2_cache = dict()

    def update_state(self):
        self.t+=1

    def optimize_model(self, model, grad_params):
        lr_t = self.learning_rate * (np.sqrt(1.0-self.beta2**self.t) / (1.0-self.beta1**self.t))
        for p in grad_params.keys():
            grad_p = grad_params[p]
            # clip gradient
            if self.grad_clip > 0:
                grad_p = np.minimum(grad_p, self.grad_clip)
                grad_p = np.maximum(grad_p, -self.grad_clip)
            grad_p_cache = self.grad_cache.get(p, np.zeros(grad_p.shape))
            grad2_p_cache = self.grad2_cache.get(p, np.zeros(grad_p.shape))
            grad_p_cache = self.beta1*grad_p_cache + (1.0-self.beta1)*grad_p
            grad2_p_cache = self.beta2*grad2_p_cache + (1.0-self.beta2)*grad_p**2

            dx_p = -lr_t * grad_p_cache / (np.sqrt(grad2_p_cache) + self.smooth_eps)
            self.grad_cache[p] = grad_p_cache
            self.grad2_cache[p] = grad2_p_cache
            model.params[p] += dx_p

sgd = SGD
rmsprop = RMSProp
adagrad = Adagrad
adadelta = Adadelta
adam = Adam