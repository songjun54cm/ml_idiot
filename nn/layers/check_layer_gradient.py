__author__ = 'SongJun-Dell'
import argparse
from Attention import Attention
from FullConnect import FullConnect
from StackFullConnect import StackFullConnect
import numpy as np

def check(get_cost_func, layer, input_data, gth_out):
    status_types = list()
    status_counts = list()
    status_params = list()

    status_types.append('PASS')
    status_counts.append(0)
    status_params.append(list())

    status_types.append('VAL SMALL')
    status_counts.append(0)
    status_params.append(list())

    status_types.append('WARNING')
    status_counts.append(0)
    status_params.append(list())

    status_types.append('ERROR')
    status_counts.append(0)
    status_params.append(list())

    warning_thr = 1e-2
    error_thr = 1.0
    delta = 1e-8

    detail = True
    cost, grad_params = get_cost_func(layer, input_data, gth_out, mode='train')
    layer_params = layer.get_params()

    for p in layer_params.keys():
        status = 0
        param_mat = layer_params[p]
        print 'checking gradient on parameter %s of shape %s.' % (p, str(param_mat.shape))
        assert param_mat.shape == grad_params[p].shape, 'Error, dims do not match: %s and %s.' % \
                                                        (str(param_mat.shape), str(grad_params[p].shape))
        for i in xrange(param_mat.size):
            param_mat.flat[i] += delta
            new_cost = get_cost_func(layer, input_data, gth_out, mode='test')
            param_mat.flat[i] -= delta

            grad_numerical = (new_cost-cost)/delta
            grad_analytic = grad_params[p].flat[i]
            if grad_analytic==0 and grad_numerical==0:
                rel_error = 0
                status = max(status, 0) # pass
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0
                status = max(status, 1) # vel small
            else:
                rel_error = abs(grad_analytic-grad_numerical) / abs(grad_numerical+grad_analytic)
                if rel_error > error_thr:
                    status = max(status, 3)
                elif rel_error > warning_thr:
                    status = max(status, 2)
                else:
                    status = max(status, 0)
            if status >= 0 and detail:
                print '%s checking param %s index %8d (val = %+8f), analytic= %+8f, numerical= %+8f, relative error= %+8f' \
                    % (status_types[status], p, i, param_mat.flat[i], grad_analytic, grad_numerical, rel_error)

        print '[%s] check gradient on parameter %s of shape %s.' % (status_types[status], p, str(param_mat.shape))
        status_counts[status] += 1
        status_params[status].append(p)

    for i in xrange(len(status_types)):
        print '%s parameters: %d' % (status_types[i], status_counts[i])
        print '\n'.join(status_params[i])

def get_cost(layer, input_data, gth_out, mode='train'):
    layer_out, layer_cache = layer.gdc_activate(input_data)
    cost = 0.5 * np.sum((layer_out - gth_out)**2)

    if mode=='train':
        grad_out = layer_out - gth_out
        grad_params = layer.gdc_backward(grad_out, layer_cache)
        return cost, grad_params
    elif mode=='test':
        return cost

def get_layer(params):
    if params['Layer_name'] == 'Attention':
        layer_class = Attention
    elif params['Layer_name'] == 'FullConnect':
        layer_class = FullConnect
    elif params['Layer_name'] == 'StackFullConnect':
        layer_class = StackFullConnect
    elif params['Layer_name'] == 'RNN':
        from RNN import RNN
        layer_class = RNN
    elif params['Layer_name'] == 'GRU':
        from GRU import GRU
        layer_class = GRU

    gdc_data = layer_class.gdc_data()
    layer = layer_class(gdc_data['layer_state'])
    input_data = gdc_data['input_data']
    gth_out = gdc_data['gth_out']
    return layer, input_data, gth_out

def main(params):
    layer, input_data, gth_out = get_layer(params)
    check(get_cost, layer, input_data, gth_out)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--Layer', dest='Layer_name', type=str, default='GRU', help='FullConnect/LSTM/Attention/StackFullConnect/RNN/GRU')

    args = parser.parse_args()
    params = vars(args)
    main(params)
