__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function

class FullConnect(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """
        state = {
            'activation_func': ,
            'input_size': ,
            'output_size':
        }
        """
        super(FullConnect, self).__init__(state, rng)
        self.activation = get_action_function(state['activation_func'])
        self.grad_act = get_gradient_function(state['activation_func'])
        self.create_variables()

    def create_variables(self):
        # self.W = add_to_params(self.params, init_matrix((self.state['input_size'], self.state['output_size']),rng=self.rng, name=self.get_variable_name('W')))
        self.W, self.W_name = self.add_params((self.state['input_size'], self.state['output_size']), 'W')
        self.regularize_param_names.append(self.W_name)

        self.b, self.b_name = self.add_params((1, self.state['output_size']), 'b')

    # def init_grads(self):
    #     grad_params = dict()
    #     self.add_grad_param(grad_params, np.zeros(self.W.shape), 'W')
    #     self.add_grad_param(grad_params, np.zeros(self.b.shape), 'b')
    #     return grad_params

    def activate(self, input_x):
        out_vec = micro_activate(input_x, self.W, self.b, self.activation)
        return out_vec

    def forward(self, input_x):
        return self.activate(input_x)

    def backward(self, grad_params, in_vecs, grad_out):
        grad_params[self.W_name] += in_vecs.transpose().dot(grad_out)
        grad_params[self.b_name] += np.sum(grad_out, axis=0, keepdims=True)
        grad_in_vecs = grad_out.dot(self.W.transpose())
        return grad_in_vecs

    """
    def backward(self, grad_params, in_vecs, out_vecs, grad_out):
        grad_out = grad_out * self.grad_act(out_vecs)
        grad_params[self.W_name] += in_vecs.transpose().dot(grad_out)
        grad_params[self.b_name] += np.sum(grad_out, axis=0, keepdims=True)
        grad_in_vecs = grad_out.dot(self.W.transpose())
        return grad_in_vecs
    """
    @staticmethod
    def gdc_data():
        input_size = 2
        x_num = 100
        output_size = 3
        layer_state = {
            'layer_name': 'full',
            'input_size': input_size,
            'output_size': output_size,
            'activation_func': 'tanh'
        }
        input_x = np.random.RandomState(1).rand(x_num, input_size)
        gth_out = np.random.RandomState(2).rand(x_num, output_size)

        gdc_data = {
            'layer_state': layer_state,
            'input_data': {'input_x':input_x},
            'gth_out': gth_out
        }
        return gdc_data

    def gdc_activate(self, input_data):
        input_x = input_data['input_x']

        layer_out = self.activate(input_x)
        cache = {
            'layer_out': layer_out,
            'input_x': input_x
        }
        return layer_out, cache

    def gdc_backward(self, grad_out, cache):
        grad_params = dict()
        for p in self.params.keys():
            grad_params[p] = np.zeros(self.params[p].shape)
        input_x = cache['input_x']
        # layer_out = cache['layer_out']
        self.backward(grad_params, input_x, grad_out)
        return grad_params
