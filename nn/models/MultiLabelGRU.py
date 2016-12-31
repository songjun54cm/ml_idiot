__author__ = 'SongJun-Dell'
from MultiLabelRNN import MultiLabelRNN
from ml_idiot.nn.layers.FullConnect import FullConnect
from ml_idiot.nn.layers.GRU import GRU


class MultiLabelGRU(MultiLabelRNN):
    def __init__(self, state):
        super(MultiLabelGRU, self).__init__(state)
        self.init_model(state)

    def init_model(self, state):
        self.encode_layer = self.add_layer(FullConnect(state['encode_layer_state']))
        self.rnn_layer = self.add_layer(GRU(state['rnn_layer_state']))
        self.decode_layer = self.add_layer(FullConnect(state['decode_layer_state']))
        self.check()

    @staticmethod
    def fullfillstate(state):
        state = MultiLabelRNN.fullfillstate(state)
        state['rnn_layer_state'].update({
            'layer_name': 'gru_layer'
        })
        state['rnn_layer_state']['gru_input_size'] = state['rnn_layer_state'].pop('rnn_input_size')
        return state