__author__ = 'JunSong'
import numpy as np
from BasicLayer import BasicLayer, get_action_function, get_gradient_function, get_grad_softmax

class SelectionLayer(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """
        :param state: = {
            'selection_type':,
            'activation_func':,
            'input_size:',
            'output_size',
         }
        :param rng:
        :return:
        """
        super(SelectionLayer, self).__init__(state, rng)
        self.activation = get_action_function(state['activation_func'])
        self.grad_act = get_gradient_function(state['activation_func'])
        self.create_variables()

    def create_variables(self):
        self.W, self.W_name = self.add_params((self.state['output_size'], self.state['input_size']), 'W')
        if not self.state['activation_func'] == 'maxhot':
            self.regularize_param_names.append(self.W_name)

    def update(self, layer):
        self.W.setfield(layer.W, dtype=self.W.dtype)

    def activate(self, input_x):
        pass

    def forward(self, input_x):
        n = input_x.shape[0]
        d1 = input_x.shape[1]
        d2 = self.state['output_size']
        if self.state['selection_type'] == 'distance':
            # x2 = np.tile(np.sum(input_x**2, axis=1, keepdims=True), [1, self.state['output_size']])
            # y2 = np.tile(np.sum(self.W**2, axis=0, keepdims=True), [input_x.shape[0], 1])
            # out_vecs = self.activation( 2.0/(x2 + y2 - 2*input_x.dot(self.W)) )
            x2 = np.tile(input_x, [1,1,d2]).reshape((n,d2,d1))
            y2 = np.tile(np.expand_dims(self.W, axis=0), [n,1,1])
            x_y = x2-y2
            out_vecs = self.activation(-0.5 * np.sum(x_y**2, axis=2))
            cache = {
                'input_x': input_x,
                'layer_out': out_vecs,
                'x_y': x_y
            }
        else:
            raise StandardError('selection type error!')
        return out_vecs, cache

    def backward(self, grad_params, grad_out, cache):
        if self.state['selection_type'] == 'distance':
            if self.state['activation_func'] == 'maxhot':
                return np.zeros(cache['input_x'].shape)
            else:
                out_vecs = cache['layer_out']
                x_y = cache['x_y']
                if self.state['activation_func'] == 'softmax':
                    tmp_grad = get_grad_softmax(out_vecs, grad_out)
                else:
                    tmp_grad = grad_out*self.grad_act(out_vecs, grad_z=grad_out)
                tmp_grad2 = np.tile(np.expand_dims(tmp_grad, axis=-1), [1,1,self.state['input_size']])
                # g = 2.0/(x_y**3)
                # grad_params[self.W_name] += np.sum(tmp_grad2 *g, axis=0)
                # grad_in_vecs = np.sum(tmp_grad2*(-g), axis=1)
                grad_params[self.W_name] += np.sum(tmp_grad2 * x_y, axis=0)
                grad_in_vecs = np.sum(tmp_grad2 * (-x_y), axis=1)
        else:
            raise StandardError('selection type error!')
        return grad_in_vecs

    @staticmethod
    def gdc_data():
        input_size = 2
        x_num = 100
        output_size = 3
        layer_state = {
            'layer_name': 'selection_layer',
            'input_size': input_size,
            'output_size': output_size,
            'activation_func': 'softmax',
            'selection_type': 'distance'
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

        layer_out, layer_cache = self.forward(input_x)
        cache = {
            'layer_out': layer_out,
            'input_x': input_x,
            'layer_cache': layer_cache
        }
        return layer_out, cache

    def gdc_backward(self, grad_out, cache):
        grad_params = dict()
        for p in self.params.keys():
            grad_params[p] = np.zeros(self.params[p].shape)
        layer_cache = cache['layer_cache']
        grad_in_vecs = self.backward(grad_params, grad_out, layer_cache)
        grad_params['input_x'] = grad_in_vecs
        return grad_params