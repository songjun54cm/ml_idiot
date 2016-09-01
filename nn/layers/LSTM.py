__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function

class LSTM(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """

        :param state: {
            'layer_name':
            'lstm_input_size':
            'hidden_size':
            'gate_act':
            'hidden_act':
            '
         }
        :param rng:
        :return:
        """
        super(LSTM, self).__init__(state, rng)

    def create_variables(self):
        # lstm input gate
        self.Win, self.Win_name = self.add_params((self.state['lstm_input_size'], self.state['hidden_size']), 'Win')
        self.regularize_param_names.append(self.Win_name)
        self.bin, self.bin_name = self.add_params((1, self.state['hidden_size']), 'bin')
        self.in_act = get_action_function(self.state.get('gate_act', 'sigmoid'))
        self.in_grad_act = get_gradient_function(self.state.get('gate_act', 'sigmoid'))

        # lstm forget gate
        self.Wfo, self.Wfo_name = self.add_params((self.state['lstm_input_size'], self.state['hidden_size']), 'Wfo')
        self.regularize_param_names.append(self.Wfo_name)
        self.bfo, self.bfo_name = self.add_params((1, self.state['hidden_size']), 'bfo')
        self.fo_act = get_action_function(self.state.get('gate_act', 'sigmoid'))
        self.fo_grad_act = get_gradient_function(self.state.get('gate_act', 'sigmoid'))

        # lstm output gate
        self.Wou, self.Wou_name = self.add_params((self.state['lstm_input_size'], self.state['hidden_size']), 'Wou')
        self.regularize_param_names.append(self.Wou_name)
        self.bou, self.bou_name = self.add_params((1, self.state['hidden_size']), 'bou')
        self.ou_act = get_action_function(self.state.get('gate_act', 'sigmoid'))
        self.ou_grad_act = get_gradient_function(self.state.get('gate_act', 'sigmoid'))

        # lstm hidden weights
        self.Wh, self.Wh_name = self.add_params((self.state['lstm_input_size'], self.state['hidden_size']), 'Wh')
        self.regularize_param_names.append(self.Wh_name)
        # lstm hidden bias
        self.bh, self.bh_name = self.add_params((1, self.state['hidden_size']), 'bh')
        self.h_act = get_action_function(self.state.get('hidden_act', 'tanh'))
        self.h_grad_act = get_gradient_function(self.state.get('hidden_act', 'tanh'))

        # store the memory cell in first n spots, and store the current output in the next n spots
        self.default_out, self.default_out_name =self.add_params((1, self.state['hidden_size']),'default_out')
        self.default_cell, self.default_cell_name = self.add_params((1, self.state['hidden_size']),'default_cell')

        self.cell_act = get_action_function('tanh')
        self.cell_grad_act = get_gradient_function('tanh')

    # def init_grads(self):
    #     grad_params = dict()
    #     self.add_grad_param(grad_params, np.zeros(self.Win.shape), 'Win')
    #     self.add_grad_param(grad_params, np.zeros(self.bin.shape), 'bin')
    #     self.add_grad_param(grad_params, np.zeros(self.Wfo.shape), 'Wfo')
    #     self.add_grad_param(grad_params, np.zeros(self.bfo.shape), 'bfo')
    #     self.add_grad_param(grad_params, np.zeros(self.Wou.shape), 'Wou')
    #     self.add_grad_param(grad_params, np.zeros(self.bou.shape), 'bou')
    #     self.add_grad_param(grad_params, np.zeros(self.Wh.shape), 'Wh')
    #     self.add_grad_param(grad_params, np.zeros(self.bh.shape), 'bh')
    #     self.add_grad_param(grad_params, np.zeros(self.default_out.shape), 'default_out')
    #     self.add_grad_param(grad_params, np.zeros(self.default_cell.shape), 'default_cell')
    #     return grad_params

    def update(self, layer):
        self.Win.setfield(layer.Win, dtype=self.Win.dtype)
        self.bin.setfield(layer.bin, dtype=self.bin.dtype)
        self.Wfo.setfield(layer.Wfo, dtype=self.Wfo.dtype)
        self.bfo.setfield(layer.bfo, dtype=self.bfo.dtype)
        self.Wou.setfield(layer.Wou, dtype=self.Wou.dtype)
        self.bou.setfield(layer.bou, dtype=self.bou.dtype)
        self.Wh.setfield(layer.Wh, dtype=self.Wh.dtype)
        self.bh.setfield(layer.bh, dtype=self.bh.dtype)
        self.default_out.setfield(layer.default_out, dtype=self.default_out.dtype)
        self.default_cell.setfield(layer.default_cell, dtype=self.default_cell.dtype)

    def activate(self, current_input_vecs, prev_out, prev_cell):
        # get memory cell and previous hidden output
        in_vecs = np.hstack([prev_out, current_input_vecs])

        # forward lstm
        input_gate = micro_activate(in_vecs, self.Win, self.bin, self.in_act)
        forget_gate = micro_activate(in_vecs, self.Wfo, self.bfo, self.fo_act)
        out_gate = micro_activate(in_vecs, self.Wou, self.bou, self.ou_act)
        hidden_vec = micro_activate(in_vecs, self.Wh, self.bh, self.h_act)

        # current memory cell
        cur_c = forget_gate * prev_cell + hidden_vec * input_gate
        # current hidden output vector
        cur_h = out_gate * self.cell_act(cur_c)

        cache = {
            'cell': cur_c,
            'lstm_in_vec': in_vecs,
            'prev_cell': prev_cell,
            'input_gate': input_gate,
            'forget_gate': forget_gate,
            'out_gate': out_gate,
            'hidden_vec': hidden_vec
        }
        return cur_h, cur_c, cache

    def forward_sequence(self, lstm_in_vecs):
        if len(lstm_in_vecs.shape) != 2: raise StandardError('error lstm in vecs shape')
        lstm_caches = list()
        num = lstm_in_vecs.shape[0]
        lstm_prev_out = self.default_out
        lstm_prev_cell = self.default_cell

        hidden_out = np.empty((num, self.state['hidden_size']))
        for i in xrange(lstm_in_vecs.shape[0]):
            lstm_prev_out, lstm_prev_cell, cache = self.activate(lstm_in_vecs[i:i+1,:], lstm_prev_out, lstm_prev_cell)
            lstm_caches.append(cache)
            hidden_out[i:i+1,:] = lstm_prev_out
        return hidden_out, lstm_caches

    def backward_whole_sequence(self, grad_params, grad_hidden_out_vecs, caches, grad_cells=None):
        """
        :param grad_hidden_out_vecs: (sequence_size, hidden_size)
        :param caches:
        :return:
        """
        if grad_cells is None:
            grad_cells = np.zeros((len(caches), self.state['hidden_size']))
        # grad_prev_cell = np.zeros(self.default_cell.shape)
        grad_recurrent_in_vecs = np.zeros((len(caches), self.state['lstm_input_size']-self.state['hidden_size']))
        for ci in reversed(xrange(len(caches))):
            cache = caches[ci]
            grad_in_vec, grad_prev_cell = self.backward_step(grad_params, grad_hidden_out_vecs[ci:ci+1,:], cache, grad_cells[ci:ci+1,:])
            if ci>0:
                grad_hidden_out_vecs[ci-1:ci,:] += grad_in_vec[:, :self.state['hidden_size']]
                grad_cells[ci-1:ci,:] += grad_prev_cell
            else:
                grad_init_hidden_out = grad_in_vec[:, :self.state['hidden_size']]
                grad_params[self.default_cell_name] += grad_prev_cell
                grad_params[self.default_out_name] += grad_init_hidden_out
            grad_recurrent_in_vec = grad_in_vec[:, self.state['hidden_size']:]
            grad_recurrent_in_vecs[ci:ci+1, :] = grad_recurrent_in_vec
        return grad_recurrent_in_vecs

    def backward_step(self, grad_params, grad_hidden_out, cache, grad_cell=None):
        cell = cache['cell']
        lstm_in_vec = cache['lstm_in_vec']
        prev_cell = cache['prev_cell']
        grad_out_gate = self.cell_act(cell) * grad_hidden_out
        if grad_cell is None: grad_cell = np.zeros(cell.shape)
        grad_cell += self.cell_grad_act(self.cell_act(cell)) * (cache['out_gate'] * grad_hidden_out)
        if prev_cell is not None:
            grad_fo_gate = prev_cell * grad_cell
            grad_prev_cell = cache['forget_gate'] * grad_cell
        else:
            grad_fo_gate = np.zeros(grad_out_gate.shape)
            grad_prev_cell = None
        grad_in_gate = cache['hidden_vec'] * grad_cell
        grad_hidden_vec = cache['input_gate'] * grad_cell

        grad_in_gate *= self.in_grad_act(cache['input_gate'])
        grad_fo_gate *= self.fo_grad_act(cache['forget_gate'])
        grad_out_gate *= self.ou_grad_act(cache['out_gate'])
        grad_hidden_vec *= self.h_grad_act(cache['hidden_vec'])

        grad_params[self.Win_name] += np.outer(lstm_in_vec, grad_in_gate)
        grad_params[self.bin_name] += np.sum(grad_in_gate, axis=0)
        grad_params[self.Wfo_name] += np.outer(lstm_in_vec, grad_fo_gate)
        grad_params[self.bfo_name] += np.sum(grad_fo_gate, axis=0)
        grad_params[self.Wou_name] += np.outer(lstm_in_vec, grad_out_gate)
        grad_params[self.bou_name] += np.sum(grad_out_gate, axis=0)
        grad_params[self.Wh_name] += np.outer(lstm_in_vec, grad_hidden_vec)
        grad_params[self.bh_name] += np.sum(grad_hidden_vec, axis=0)

        grad_in_vec = grad_in_gate.dot(self.Win.transpose())
        grad_in_vec += grad_fo_gate.dot(self.Wfo.transpose())
        grad_in_vec += grad_out_gate.dot(self.Wou.transpose())
        grad_in_vec += grad_hidden_vec.dot(self.Wh.transpose())

        return grad_in_vec, grad_prev_cell

    @staticmethod
    def gdc_data():
        x_size = 2
        hidden_size =3
        input_size = x_size + hidden_size
        x_num = 10
        layer_state = {
            'layer_name': 'gru',
            'x_size': x_size,
            'lstm_input_size': input_size,
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
        grad_in_vecs = self.backward_whole_sequence(grad_params, grad_out, layer_cache)
        grad_params['input_x'] = grad_in_vecs
        return grad_params