__author__ = 'JunSong'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function

class RNN(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """
        state = {
            'rnn_input_size': ,
            'hidden_size': ,
            'h_act': ,
        }
        """
        super(RNN, self).__init__(state, rng)

    def create_variables(self):
        # rnn input gate
        self.Wh, self.Wh_name = self.add_params((self.state['rnn_input_size'], self.state['hidden_size']), 'Wh')
        self.regularize_param_names.append(self.Wh_name)
        self.bh, self.bh_name = self.add_params((1, self.state['hidden_size']), 'bh')
        self.h_act = get_action_function(self.state.get('h_act', 'tanh'))
        self.h_grad_act = get_gradient_function(self.state.get('h_act', 'tanh'))

        """
        # rnn output gate
        self.Wou, self.Wou_name = self.add_params((self.state['hidden_size'], self.state['rnn_output_size']), 'Wou')
        self.regularize_param_names.append(self.Wou_name)
        self.bou, self.bou_name = self.add_params((1, self.state['rnn_output_size']), 'bou')
        self.ou_act = get_action_function(self.state.get('ou_act', 'tanh'))
        self.ou_grad_act = get_gradient_function(self.state.get('ou_act', 'tanh'))
        """
        self.default_hidden, self.default_hidden_name = self.add_params((1, self.state['hidden_size']), 'default_hidden')

    def activate(self, current_input_vecs, prev_hidden):
        in_vecs = np.hstack([prev_hidden, current_input_vecs])
        hidden_vec = micro_activate(in_vecs, self.Wh, self.bh, self.h_act)
        out_vec = hidden_vec
        cache = {
            'rnn_in_vecs': in_vecs,
            'hidden_vec': hidden_vec
        }
        return out_vec, cache

    def forward_sequence(self, rnn_in_vecs):
        if len(rnn_in_vecs.shape) !=2: raise StandardError('error rnn in vecs shape')
        rnn_caches = list()
        num = rnn_in_vecs.shape[0]
        rnn_prev_hidden = self.default_hidden

        hidden_out = np.empty((num, self.state['hidden_size']))
        for i in xrange(rnn_in_vecs.shape[0]):
            rnn_prev_hidden, cache = self.activate(rnn_in_vecs[i:i+1,:], rnn_prev_hidden)
            rnn_caches.append(cache)
            hidden_out[i:i+1,:] = rnn_prev_hidden

        return hidden_out, rnn_caches

    def backward_whole_sequence(self, grad_params, grad_hidden_out_vecs, caches):
        grad_recurrent_in_vecs = np.zeros((len(caches), self.state['rnn_input_size']-self.state['hidden_size']))
        for ci in reversed(xrange(len(caches))):
            cache = caches[ci]
            grad_in_vec = self.backward_step(grad_params, grad_hidden_out_vecs[ci:ci+1,:], cache)
            if ci>0:
                grad_hidden_out_vecs[ci-1:ci,:] += grad_in_vec[:,:self.state['hidden_size']]
            else:
                grad_init_hidden_out = grad_in_vec[:, :self.state['hidden_size']]
                grad_params[self.default_hidden_name] += grad_init_hidden_out
            grad_recurrent_in_vec = grad_in_vec[:, self.state['hidden_size']:]
            grad_recurrent_in_vecs[ci:ci+1, :] = grad_recurrent_in_vec

        return grad_recurrent_in_vecs

    def backward_step(self, grad_params, grad_hidden_out, cache):
        rnn_in_vec = cache['rnn_in_vecs']
        hidden_vec = cache['hidden_vec']
        grad_hidden = grad_hidden_out * self.h_grad_act(hidden_vec)
        grad_params[self.Wh_name] += np.outer(rnn_in_vec, grad_hidden)
        grad_params[self.bh_name] += np.sum(grad_hidden, axis=0)
        grad_in_vec = grad_hidden.dot(self.Wh.transpose())
        return grad_in_vec

    @staticmethod
    def gdc_data():
        x_size = 2
        hidden_size =3
        input_size = x_size + hidden_size
        x_num = 10
        layer_state = {
            'layer_name': 'rnn',
            'x_size': x_size,
            'rnn_input_size': input_size,
            'hidden_size': hidden_size,
            'h_act': 'tanh'
        }
        input_x = np.random.RandomState(1).rand(x_num, x_size)
        gth_out = np.random.RandomState(2).rand(x_num, hidden_size)

        gdc_data = {
            'layer_state': layer_state,
            'input_data': {'input_x': input_x},
            'gth_out': gth_out,
        }
        return gdc_data

    def gdc_activate(self, input_data):
        input_x = input_data['input_x']
        layer_out, layer_cache = self.forward_sequence(input_x)
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
        input_x = cache['input_x']
        layer_cache = cache['layer_cache']
        self.backward_whole_sequence(grad_params, grad_out, layer_cache)
        return grad_params