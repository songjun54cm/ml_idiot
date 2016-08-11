__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function
from FullConnect import FullConnect

class StackFullConnect(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(StackFullConnect, self).__init__(state, rng)
        self.layer_sequence = list()
        self.num_inner_layers = len(state['inner_states'])
        for i,layer_state in enumerate(state['inner_states']):
            layer_state['layer_name'] = state['layer_name'] + '_stack_' + str(i)
            self.stack_layer(FullConnect(layer_state))

    def create_variables(self):
        pass

    def stack_layer(self, layer):
        self.layer_sequence.append(layer)
        self.params.update(layer.params)
        self.regularize_param_names += layer.regularize_param_names

    def forward(self, input_x):
        inner_hidden_outs = list()
        inner_input_x = input_x
        for inner_layer in self.layer_sequence:
            inner_out = inner_layer.forward(inner_input_x)
            inner_hidden_outs.append(inner_out)
            inner_input_x = inner_out
        out_vec = inner_out
        cache = {
            'inner_hidden_outs': inner_hidden_outs[:-1]
        }
        return out_vec, cache

    def backward(self, grad_params, in_vecs, grad_out, cache):
        grad_inner_out = grad_out
        for li in reversed(xrange(self.num_inner_layers)):
            if li == 0:
                layer_in = in_vecs
            else:
                layer_in = cache['inner_hidden_outs'][li-1]
            inner_layer = self.layer_sequence[li]
            grad_inner_out = inner_layer.backward(grad_params, layer_in, grad_inner_out)
        return grad_inner_out

    @staticmethod
    def gdc_data():
        input_size = 2
        x_num = 100
        output_size = 3
        state = {
            'layer_name': 'stack_full',
            'inner_states':[
                {'input_size': input_size, 'output_size': 3, 'activation_func': 'tanh'},
                {'input_size': 3, 'output_size': 2, 'activation_func': 'tanh'},
                {'input_size': 2, 'output_size': 3, 'activation_func': 'tanh'}
            ]
        }
        input_x = np.random.RandomState(1).rand(x_num, input_size)
        gth_out = np.random.RandomState(2).rand(x_num, output_size)

        gdc_data = {
            'layer_state': state,
            'input_data': {'input_x': input_x},
            'gth_out': gth_out
        }
        return gdc_data

    def gdc_activate(self, input_data):
        input_x = input_data['input_x']
        layer_out, cache = self.forward(input_x)
        gdc_cache = {
            'layer_out': layer_out,
            'forward_cache': cache,
            'input_x': input_x
        }
        return layer_out, gdc_cache

    def gdc_backward(self, grad_out, gdc_cache):
        grad_params = dict()
        for p in self.params.keys():
            grad_params[p] = np.zeros(self.params[p].shape)
        input_x = gdc_cache['input_x']
        forward_cache = gdc_cache['forward_cache']
        self.backward(grad_params, input_x, grad_out, forward_cache)
        return grad_params
