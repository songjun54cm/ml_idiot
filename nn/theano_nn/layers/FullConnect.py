__author__ = 'JunSong'
import numpy as np
from BasicLayer import BasicLayer, add_to_params, create_shared, micro_activate, get_activation_function

class FullConnect(BasicLayer):
    """
    Base object for neural network layers.

    A layer has an input set of neurons, and
    a hidden activation. The activation, f, is a
    function applied to the affine transformation
    of x by the connection matrix W, and the bias
    vector b.

    > y = f ( W * x + b )
    """
    def __init__(self, state, rng=np.random.RandomState(1234)):
        """
        :param state: {
            'input_size':
            'output_size':
            'activation_func':
            'layer_name':
        }
        :return:
        """
        super(FullConnect, self).__init__(state)
        self.activation = get_activation_function(state['activation_func'])
        self.create_variables()

    def create_variables(self):
        # create weight matrix and bias
        self.W, self.W_name = self.add_params((self.state['input_size'], self.state['output_size']), 'W')
        self.regularize_params.append(self.W)
        self.b, self.b_name = self.add_params((1, self.state['output_size']), 'b')

    def forward(self, input_x):
        out_vec = micro_activate(input_x, self.W, self.b, self.activation)
        return out_vec