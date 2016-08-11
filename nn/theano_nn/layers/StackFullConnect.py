__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, add_to_params, create_shared, micro_activate, get_activation_function
from FullConnect import FullConnect
class StackFullConnect(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(StackFullConnect, self).__init__(state, rng)
        self.layer_sequence = list()
        self.num_inner_layer = len(state['inner_states'])
        for i,layer_state in enumerate(state['inner_states']):
            layer_state['layer_name'] = state['layer_name'] + '_stack_' + str(i)
            self.stack_layer(FullConnect(layer_state))

    def stack_layer(self, layer):
        self.layer_sequence.append(layer)
        self.params += layer.params
        self.regularize_params += layer.regularize_params

    def forward(self, input_x):
        out_vec = input_x
        for layer in self.layer_sequence:
            out_vec = layer.forward(out_vec)
        return out_vec
