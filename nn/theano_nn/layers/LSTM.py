__author__ = 'JunSong'
from BasicLayer import BasicLayer, add_to_params, create_shared, micro_activate
import numpy as np
import theano
import theano.tensor as T

class LSTM(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """

        :param state:{
            layer_name, # string
            lstm_input_size, # number
            hidden_size, # number
            gate_act, # function
            hidden_act, # function
         }
        :param rng:
        :return:
        """
        super(LSTM, self).__init__(state)
        self.create_variables()

    def create_variables(self):
        # lstm input gate
        self.Win = add_to_params(self.params, create_shared((self.state['lstm_input_size'], self.state['hidden_size']), self.rng, name=self.get_variable_name('Win')))
        self.regularize_params.append(self.Win)
        self.bin = add_to_params(self.params, create_shared((1, self.state['hidden_size']), self.rng, name=self.get_variable_name('bin')))
        # lstm forget gate
        self.Wfo = add_to_params(self.params, create_shared((self.state['lstm_input_size'], self.state['hidden_size']), self.rng, name=self.get_variable_name('Wfo')))
        self.regularize_params.append(self.Wfo)
        self.bfo = add_to_params(self.params, create_shared((1, self.state['hidden_size']), self.rng, name=self.get_variable_name('bfo')))
        # lstm output gate
        self.Wou = add_to_params(self.params, create_shared((self.state['lstm_input_size'], self.state['hidden_size']), self.rng, name=self.get_variable_name('Wou')))
        self.regularize_params.append(self.Wou)
        self.bou = add_to_params(self.params, create_shared((1, self.state['hidden_size']), self.rng, name=self.get_variable_name('bou')))
        # lstm hidden weights
        self.Wh = add_to_params(self.params, create_shared((self.state['lstm_input_size'], self.state['hidden_size']), self.rng, name=self.get_variable_name('Wh')))
        self.regularize_params.append(self.Wh)
        # lstm hidden bias
        self.bh = add_to_params(self.params, create_shared((1, self.state['hidden_size']), self.rng, name=self.get_variable_name('bh')))
        # default memory cell
        self.default_cell = add_to_params(self.params, theano.shared(np.zeros((1, self.state['hidden_size'])), name=self.get_variable_name('default_cell')))
        # default output
        self.default_output = add_to_params(self.params, theano.shared(np.zeros((1, self.state['hidden_size'])), name=self.get_variable_name('default_output')))

    def initial_state_with_taps(self, num=None):
        if num is not None:
            cell = T.repeat(self.default_cell, num, axis=0)
            output = T.repeat(self.default_output, num, axis=0)
        else:
            cell = self.default_cell
            output = self.default_output
        return dict(initial=output, taps=[-1]), dict(initial=cell, taps=[-1])

    def forward(self, current_input_x, prev_out_put, prev_cell):
        # get memory cell and previous hidden output
        if current_input_x.ndim>1:
            # input to lstm
            in_vecs = T.concatenate([current_input_x, prev_out_put], axis=1)
        else:
            in_vecs = T.concatenate([current_input_x, prev_out_put])

        # forward lstm
        input_gate = micro_activate(in_vecs, self.Win, self.bin, self.state.get('gate_act', T.nnet.sigmoid))
        forget_gate = micro_activate(in_vecs, self.Wfo, self.bfo, self.state.get('gate_act', T.nnet.sigmoid))
        out_gate = micro_activate(in_vecs, self.Wou, self.bou, self.state.get('gate_act', T.nnet.sigmoid))
        hidden_vec = micro_activate(in_vecs, self.Wh, self.bh, self.state.get('hidden_act', T.tanh))

        # current memory cell
        cur_c = forget_gate * prev_cell + hidden_vec * input_gate
        # current hidden output vector
        cur_h = out_gate * T.tanh(cur_c)

        # new state of lstm
        return [cur_h, cur_c]

    def forward_whole_sequence(self, seq_in_x):
        # seq_in_x: (max_len, num_sample, hidden_size)
        def_out, def_cell = self.initial_state_with_taps(seq_in_x.shape[1])
        output_info = [def_out, def_cell]
        results, _ = theano.scan(fn=self.forward,
                                 sequences=[seq_in_x],
                                 outputs_info=output_info)
        outs, cells = results
        return outs, cells
