__author__ = 'SongJun-Dell'
import numpy as np
def identity_func(x):
    return x
def grad_identity(z):
    return np.ones(z.shape)
def tanh_func(x):
    return np.tanh(x)
def grad_tanh(z):
    return 1-z**2
def relu_func(x):
    return np.maximum(x, 0)
def grad_relu(z):
    return z>0
def sigmoid_func(x):
    return 1.0/(1.0+np.exp(-x))
def grad_sigmoid(z):
    return z*(1-z)

def sigmoid10_func(x):
    return 1.0/(1.0+np.exp(-10*x))
def grad_sigmoid10(z):
    return 10*z*(1-z)

def softmax_func(x, axis=1):
    maxes = np.amax(x, axis=axis, keepdims=True)
    e = np.exp(x - maxes)
    p = e / np.sum(e, axis=axis, keepdims=True)
    return p
def grad_softmax(z):
    # return z*(1-z)
    # z must be (N, d)
    if z.shape[0] == 1:
        g = np.diag(z[0]) - z.transpose().dot(z)
    else:
        assert(len(z.shape)==2, 'shape of z too much')
        g = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        for i in xrange(z.shape[0]):
            g[i] += grad_softmax(z[i:i+1,:])
    return g
def get_grad_softmax(z, grad_z):
    # z and grad_out must be (N, d)
    assert(z.shape==grad_z.shape, 'z.shape not equals to grad_z.shape')
    g = grad_softmax(z)
    grad = np.zeros(z.shape)
    if z.shape[0]==1:
        grad = grad_z.dot(g)
    else:
        assert(len(z.shape)==2, 'get_grad_softmax: shape of z too much')
        for i in xrange(z.shape[0]):
            grad[i:i+1,:] = grad_z[i:i+1,:].dot(g[i])
    return grad



def get_action_function(fun_name):
    """
    do the action based on the function name on the data
    :param fun_name: 'identity'/'tanh'/'relu'/'sigmoid'
    :param x: numpy.array
    :return: the result
    """
    if fun_name == 'identity':
        return identity_func
    elif fun_name == 'tanh':
        return tanh_func
    elif fun_name == 'relu':
        return relu_func
    elif fun_name == 'sigmoid':
        return sigmoid_func
    elif fun_name == 'softmax':
        return softmax_func
    elif fun_name == 'sigmoid10':
        return sigmoid10_func
    else:
        StandardError("error function name")
    return

def get_gradient_function(fun_name):
    """
    calculate the gradient of the input given the output value
    :param fun_name: 'identity'/'tanh'/'relu'/'sigmoid'
    :param z: the output value of the corresponding activation function
    :return: the gradient of x
    """
    if fun_name == 'identity':
        return grad_identity
    elif fun_name == 'tanh':
        return grad_tanh
    elif fun_name == 'relu':
        return grad_relu
    elif fun_name == 'sigmoid':
        return grad_sigmoid
    elif fun_name == 'softmax':
        return grad_softmax
    elif fun_name == 'sigmoid10':
        return grad_sigmoid10
    else:
        StandardError("error function name")
    return

def micro_activate(x, w, b, act):
    """
    :param x: matrix (N,D) or (D,), (num_sample, dimension1) or (dimension1,)
    :param w: matrix dimension1 x dimension2
    :param b: dimension2
    :param act:
    :return:
    """
    if len(x.shape) > 1:
        res = x.dot(w) + b
    else:
        res = w.transpose().dot(x) + b
    if act is not None:
        res = act(res)
    return res

def add_to_params(params, new_params, param_name):
    params[param_name] = new_params
    return new_params

def init_matrix(shape, rng=np.random.RandomState(1234), name=None, magic_number=0.1):
    return rng.standard_normal(shape) * magic_number

class BasicLayer(object):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        self.params = dict()
        self.grad_params = dict()
        self.regularize_param_names = list()
        self.state = state
        self.rng = rng
        self.create_variables()

    def create_variables(self):
        raise StandardError('Not implemented error!')

    def get_variable_name(self, name):
        return self.state['layer_name'] + '_' + name

    def add_params(self, shape, name):
        param_name = self.get_variable_name(name)
        param_values = add_to_params(self.params, init_matrix(shape, rng=self.rng), param_name=param_name)
        return param_values, param_name

    def get_params(self):
        return self.params

    def add_grad_param(self, grad_params, param_vals, name):
        param_name = self.get_variable_name(name)
        grad_params[param_name] = param_vals

    def clear_grads(self, grad_params):
        for g in grad_params.keys():
            grad_params[g].fill(0)

    def get_regularization(self):
        reg_value = 0.0
        for p in self.regularize_param_names:
            reg_value += np.linalg.norm(self.params[p])**2
        return reg_value

    def grad_regularization(self, grad_params):
        for p in self.regularize_param_names:
            grad_params[p] += self.state['regularize_rate'] * self.params[p]
        return grad_params

    def init_grads(self):
        grad_params = dict()
        for p in self.params.keys():
            grad_params[p] = np.zeros(self.params[p].shape)
        return grad_params