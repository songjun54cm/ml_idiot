__author__ = 'SongJun-Dell'
import theano
import theano.tensor as T
import numpy as np

def micro_activate(x, w, b, act):
    if x.ndim>1:
        if act is None:
            return T.dot(x,w) + T.repeat(b, x.shape[0], axis=0)
        return act(T.dot(x, w) + T.repeat(b, x.shape[0], axis=0))
    else:
        if act is None:
            res = T.dot(w.T, x) + b
        res = act(T.dot(w.T, x) + b)
        return res.flatten()

def add_to_params(params, new_params):
    params.append(new_params)
    return new_params

def NormalInit(rng, sizeX, sizeY=0, scale=0.01, sparsity=-1):
    """
    Normal Initialization
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)

    if sizeY == 0:
        return (rng.standard_normal(sizeX) * scale / sizeX).astype(theano.config.floatX)

    if sparsity < 0:
        sparsity = sizeY

    sparsity = np.minimum(sizeY, sparsity)
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    return values.astype(theano.config.floatX)

def create_shared(size, rng=np.random.RandomState(1234), name=None, magic_number=0.1):
    """
    Creates a shared matrix or vector
    using the given in_size and out_size.

    Inputs
    ------

    out_size int            : outer dimension of the
                              vector or matrix
    in_size  int (optional) : for a matrix, the inner
                              dimension.

    Outputs
    -------

    theano shared : the shared matrix, with random numbers in it

    """
    if type(size) is not tuple:
        # return theano.shared(NormalInit(rng, size), name=name)
        return theano.shared(rng.standard_normal(size)*magic_number, name=name)
    else:
        # return theano.shared(NormalInit(rng, size[0], size[1]), name=name)
        return theano.shared(rng.standard_normal((size[0], size[1]))*magic_number, name=name)

def softmax(x):
    """
    x: N x D
    Wrapper for softmax, helps with
    pickling, and removing one extra
    dimension that Theano adds during
    its exponential normalization.
    """
    return T.nnet.softmax(x)

def identity(x):
    return x

def get_activation_function(fun_name):
    if fun_name == 'softmax':
        return T.nnet.softmax
    else:
        StandardError("error function name")

class BasicLayer(object):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        self.params = list()
        self.regularize_params = list()
        self.state = state
        self.rng = rng

    def get_variable_name(self, name):
        return self.state['layer_name'] + '_' + name

    def add_params(self, shape, name):
        param_name = self.get_variable_name(name)
        param_values = create_shared(shape, rng=self.rng, name=param_name)
        self.params.append(param_values)
        return param_values, param_name
