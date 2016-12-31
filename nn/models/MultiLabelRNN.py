__author__ = 'SongJun-Dell'
from ml_idiot.nn.models.MultiLabelSeq import MultiLabelSeq
from ml_idiot.nn.layers.FullConnect import FullConnect
from ml_idiot.nn.layers.RNN import RNN
from ml_idiot.utils.loss_functions import distance_loss, grad_distance_loss

class MultiLabelRNN(MultiLabelSeq):
    def __init__(self, state):
        super(MultiLabelRNN, self).__init__(state)
        self.loss_func = distance_loss
        self.grad_loss_func = grad_distance_loss
        self.init_model(state)

    def init_model(self, state):
        self.encode_layer = self.add_layer(FullConnect(state['encode_layer_state']))
        self.rnn_layer = self.add_layer(RNN(state['rnn_layer_state']))
        self.decode_layer = self.add_layer(FullConnect(state['decode_layer_state']))
        self.check()


    @staticmethod
    def fullfillstate(state):
        encode_layer_state = {
            'layer_name': 'encode_layer',
            'activation_func': 'tanh',
            'input_size': state['input_size'],
            'output_size': state['hidden_size']
        }
        rnn_layer_state = {
            'layer_name': 'rnn_layer',
            'rnn_input_size': state['hidden_size']*2,
            'hidden_size': state['hidden_size'],
            'h_act': 'tanh'
        }
        decode_layer_state = {
            'layer_name': 'decode_layer',
            'activation_func': 'sigmoid',
            'input_size': state['hidden_size'],
            'output_size': state['label_num']
        }

        state['encode_layer_state'] = encode_layer_state
        state['rnn_layer_state'] = rnn_layer_state
        state['decode_layer_state'] = decode_layer_state
        return state

    def forward_sample(self, sample_data, mode):
        recurrent_feas = sample_data['recurrent_feas']
        # target_feas = sample_data['target_feas']
        encode_out, encode_cache = self.encode_layer.forward(recurrent_feas)
        rnn_out, recurrent_cache = self.rnn_layer.forward_sequence(encode_out)
        decode_out, decode_cache = self.decode_layer.forward(rnn_out)
        forward_cache = {
            'encode_in': recurrent_feas,
            'encode_cache': encode_cache,
            'encode_out': encode_out,
            'rnn_out': rnn_out,
            'rnn_caches': recurrent_cache,
            'decode_out': decode_out,
            'decode_cache': decode_cache,
            'target_feas': sample_data['target_feas']
        }
        return decode_out, forward_cache

    def backward_sample(self, grad_params, forward_sample_cache):
        target_feas = forward_sample_cache['target_feas']
        decode_out = forward_sample_cache['decode_out']
        decode_cache = forward_sample_cache['decode_cache']
        rnn_out = forward_sample_cache['rnn_out']
        rnn_caches = forward_sample_cache['rnn_caches']
        encode_in = forward_sample_cache['encode_in']
        encode_cache = forward_sample_cache['encode_cache']

        grad_decode_out = grad_distance_loss(decode_out, target_feas)
        grad_rnn_out = self.decode_layer.backward(grad_params, grad_decode_out, decode_cache)
        grad_encode_out = self.rnn_layer.backward_whole_sequence(grad_params, grad_rnn_out, rnn_caches)
        grad_in_vecs = self.encode_layer.backward(grad_params, grad_encode_out, encode_cache)