__author__ = 'JunSong'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function

class GRU(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """

        :param state:{
            'layer_name':
            'gru_input_size':
            'hidden_size':,
            'gate_act':
            'hidden_act':
         }
        :param rng:
        :return:
        """
        super(GRU, self).__init__(state, rng)

    def create_variables(self):
        # GRU update gate
        self.Wup, self.Wup_name = self.add_params((self.state['gru_input_size'], self.state['hidden_size']), 'Wup')
        self.regularize_param_names.append(self.Wup_name)
        self.bup, self.bup_name = self.add_params((1, self.state['hidden_size']), 'bup')
        self.up_act = get_action_function(self.state.get('gate_act', 'sigmoid'))
        self.up_grad_act = get_gradient_function(self.state.get('gate_act', 'sigmoid'))

        # GRU reset gate
        self.Wre, self.Wre_name = self.add_params((self.state['gru_input_size'], self.state['hidden_size']), 'Wre')
        self.regularize_param_names.append(self.Wre_name)
        self.bre, self.bre_name = self.add_params((1, self.state['hidden_size']), 'bre')
        self.re_act = get_action_function(self.state.get('gate_act', 'sigmoid'))
        self.re_grad_act = get_gradient_function(self.state.get('gate_act', 'sigmoid'))

        # GRU hidden weights
        self.Wh_in, self.Wh_in_name = self.add_params((self.state['gru_input_size']-self.state['hidden_size'], self.state['hidden_size']), 'Wh_in')
        self.regularize_param_names.append(self.Wh_in_name)
        self.Wh_h, self.Wh_h_name = self.add_params((self.state['hidden_size'], self.state['hidden_size']), 'Wh_h')
        self.regularize_param_names.append(self.Wh_h_name)
        self.bh, self.bh_name = self.add_params((1, self.state['hidden_size']), 'bh')
        self.h_act = get_action_function(self.state.get('hidden_act', 'tanh'))
        self.h_grad_act = get_gradient_function(self.state.get('hidden_act', 'tanh'))

        # default hidden
        self.default_out, self.default_out_name = self.add_params((1, self.state['hidden_size']),'default_out')

    def activate(self, current_in_vecs, prev_out):
        in_vecs = np.hstack([prev_out, current_in_vecs])

        update_gate = micro_activate(in_vecs, self.Wup, self.bup, self.up_act)
        reset_gate = micro_activate(in_vecs, self.Wre, self.bre, self.re_act)
        if current_in_vecs.shape>1:
            temp_hidden = self.h_act( current_in_vecs.dot(self.Wh_in) + reset_gate * prev_out.dot(self.Wh_h) + self.bh)
        else:
            temp_hidden = self.h_act(
                self.Wh_in.tranpose().dot(current_in_vecs) + reset_gate*self.Wh_in.transpose().dot(prev_out) + self.bh
            )
        hidden = update_gate * prev_out + (1.0-update_gate)*temp_hidden

        cache = {
            'gru_in_vec': in_vecs,
            'update_gate': update_gate,
            'reset_gate': reset_gate,
            'temp_hidden': temp_hidden,
            'hidden_vec': hidden,
        }
        return hidden, cache

    def forward_sequence(self, gru_in_vecs):
        if len(gru_in_vecs.shape) != 2: raise StandardError('error gru in vecs shape')
        gru_caches = list()
        num = gru_in_vecs.shape[0]
        gru_prev_out = self.default_out

        hidden_out = np.empty((num, self.state['hidden_size']))
        for i in xrange(gru_in_vecs.shape[0]):
            gru_prev_out, cache = self.activate(gru_in_vecs[i:i+1,:], gru_prev_out)
            gru_caches.append(cache)
            hidden_out[i:i+1,:] = gru_prev_out
        return hidden_out, gru_caches

    def backward_whole_sequence(self, grad_params, grad_hidden_out_vecs, caches):
        grad_recurrent_in_vecs = np.zeros((len(caches), self.state['gru_input_size']-self.state['hidden_size']))
        for ci in reversed(xrange(len(caches))):
            cache = caches[ci]
            grad_in_vec = self.backward_step(grad_params, grad_hidden_out_vecs[ci:ci+1, :], cache)
            if ci>0:
                grad_hidden_out_vecs[ci-1:ci,:] += grad_in_vec[:, :self.state['hidden_size']]
            else:
                grad_init_hidden_out = grad_in_vec[:,:self.state['hidden_size']]
                grad_params[self.default_out_name] += grad_init_hidden_out
            grad_recurrent_in_vec = grad_in_vec[:, self.state['hidden_size']:]
            grad_recurrent_in_vecs[ci:ci+1,:] = grad_recurrent_in_vec
        return grad_recurrent_in_vecs

    def backward_step(self, grad_params, grad_hidden_out, cache):
        gru_in_vec = cache['gru_in_vec']
        grad_in_vec = np.zeros(gru_in_vec.shape)
        temp_hidden = cache['temp_hidden']
        prev_hidden = gru_in_vec[:, :self.state['hidden_size']]
        reset_gate = cache['reset_gate']
        update_gate = cache['update_gate']

        grad_update_gate = (prev_hidden - temp_hidden) * grad_hidden_out
        grad_prev_hidden = update_gate * grad_hidden_out
        grad_temp_hidden = (1.0 - update_gate) * grad_hidden_out

        grad_temp_hidden *= self.h_grad_act(temp_hidden)
        grad_reset_gate =  prev_hidden.dot(self.Wh_h) * grad_temp_hidden
        grad_params[self.Wh_in_name] += np.outer(gru_in_vec[:,self.state['hidden_size']:], grad_temp_hidden)
        # grad_params[self.Wh_h_name] += np.outer(prev_hidden, grad_temp_hidden) * np.sum(cache['reset_gate'],axis=0, keepdims=True).transpose()
        grad_params[self.Wh_h_name] += np.outer(prev_hidden, grad_temp_hidden*reset_gate)
        grad_params[self.bh_name] += np.sum(grad_temp_hidden, axis=0)
        grad_curr_in_vec = grad_temp_hidden.dot(self.Wh_in.transpose())
        grad_prev_hidden += (grad_temp_hidden*reset_gate).dot(self.Wh_h.transpose())

        grad_reset_gate *= self.re_grad_act(reset_gate)
        grad_params[self.Wre_name] += np.outer(gru_in_vec, grad_reset_gate)
        grad_params[self.bre_name] += np.sum(grad_reset_gate, axis=0)
        grad_in_vec += grad_reset_gate.dot(self.Wre.transpose())

        grad_update_gate *= self.up_grad_act(update_gate)
        grad_params[self.Wup_name] += np.outer(gru_in_vec, grad_update_gate)
        grad_params[self.bup_name] += np.sum(grad_update_gate, axis=0)
        grad_in_vec += grad_update_gate.dot(self.Wup.transpose())

        grad_in_vec[:,self.state['hidden_size']:] += grad_curr_in_vec
        grad_in_vec[:,:self.state['hidden_size']] += grad_prev_hidden

        return grad_in_vec


    @staticmethod
    def gdc_data():
        x_size = 2
        hidden_size =3
        input_size = x_size + hidden_size
        x_num = 10
        layer_state = {
            'layer_name': 'gru',
            'x_size': x_size,
            'gru_input_size': input_size,
            'hidden_size': hidden_size,
            'gate_act': 'tanh',
            'hidden_act': 'tanh'

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