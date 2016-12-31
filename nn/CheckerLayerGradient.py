__author__ = 'JunSong'
import numpy as np
from CheckerGradient import CheckerGraident

class CheckerLayerGradient(CheckerGraident):
    def __init__(self, layer_class):
        super(CheckerLayerGradient, self).__init__()
        gdc_data = layer_class.gdc_data()
        layer = layer_class(gdc_data['layer_state'])
        self.input_data = gdc_data['input_data']
        self.gth_out = gdc_data['gth_out']
        self.layer = layer

    def get_loss(self, mode='train'):
        layer_out, layer_cache =self.layer.gdc_activate(self.input_data)
        loss = 0.5 * np.sum((layer_out - self.gth_out)**2)
        if mode in ['train', 'gc']:
            grad_out = layer_out - self.gth_out
            grad_params = self.layer.gdc_backward(grad_out, layer_cache)
            return loss, grad_params
        elif mode=='test':
            return loss
        else:
            raise StandardError('mode error.')

    def get_gc_params(self):
        gc_params = dict()
        gc_params.update(self.layer.get_params())
        gc_params.update(self.input_data)
        return gc_params


def get_layer_class(params):
    if params['Layer_name'] == 'Attention':
        from layers.Attention import Attention
        layer_class = Attention
    elif params['Layer_name'] == 'FullConnect':
        from layers.FullConnect import FullConnect
        layer_class = FullConnect
    elif params['Layer_name'] == 'MultiFullConnect':
        from layers.MultiFullConnect import MultiFullConnect
        layer_class = MultiFullConnect
    elif params['Layer_name'] == 'StackFullConnect':
        from layers.StackFullConnect import StackFullConnect
        layer_class = StackFullConnect
    elif params['Layer_name'] == 'RNN':
        from layers.RNN import RNN
        layer_class = RNN
    elif params['Layer_name'] == 'GRU':
        from layers.GRU import GRU
        layer_class = GRU
    elif params['Layer_name'] == 'LSTM':
        from layers.LSTM import LSTM
        layer_class = LSTM
    elif params['Layer_name'] == 'SelectionLayer':
        from layers.SelectionLayer import SelectionLayer
        layer_class = SelectionLayer
    else:
        layer_class = None
    return layer_class


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--Layer', dest='Layer_name', type=str, default='StackFullConnect',
                        help='FullConnect/MultiFullConnect/LSTM/Attention/StackFullConnect/RNN/GRU'
                             'SelectionLayer')

    args = parser.parse_args()
    params = vars(args)
    layer_class = get_layer_class(params)
    checker = CheckerLayerGradient(layer_class)
    checker.check_gradient()