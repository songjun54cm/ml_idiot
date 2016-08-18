import argparse
import numpy as np
from BasicLayer import get_action_function, get_gradient_function, get_grad_softmax
__author__ = 'SongJun-Dell'

def get_check_data():
    x_num = 2
    x_size = 3
    input_data = np.random.RandomState(1).rand(x_num, x_size)
    gth_out = np.random.RandomState(2).rand(x_num, x_size)
    return input_data, gth_out

def get_funcs(params):
    act_func = get_action_function(params['function'])
    grad_func = get_gradient_function(params['function'])
    return act_func, grad_func

def check(params, act_func, grad_func, input_data, gth_out):
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
    delta = 1e-5

    detail = True
    cost, grad_vals = get_cost(params, act_func, grad_func, input_data, gth_out, mode='train')

    status = 0
    print 'checking gradient on input data with shape %s.' % str(input_data.shape)

    for i in xrange(input_data.size):
        input_data.flat[i] += delta
        new_cost = get_cost(params, act_func, grad_func, input_data, gth_out, mode='test')
        input_data.flat[i] -= delta

        grad_numerical = (new_cost-cost)/delta
        grad_analytic = grad_vals.flat[i]
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
            print '%s checking input data index %8d (val = %+8f), analytic= %+8f, numerical= %+8f, relative error= %+8f' \
                % (status_types[status], i, input_data.flat[i], grad_analytic, grad_numerical, rel_error)

    print '[%s] check gradient on input data with shape %s.' % (status_types[status], str(input_data.shape))
    status_counts[status] += 1

    for i in xrange(len(status_types)):
        print '%s parameters: %d' % (status_types[i], status_counts[i])

def get_cost(params, act_func, grad_func, input_data, gth_out, mode):
    out = act_func(input_data)
    cost = 0.5 * np.sum((out-gth_out)**2)
    if mode == 'train':
        grad_out = out - gth_out
        if params['function'] == 'softmax':
            grad_vals = get_grad_softmax(out, grad_out)
        else:
            grad_vals = grad_out * grad_func(out)
        return cost, grad_vals
    elif mode == 'test':
        return cost

def main(params):
    input_data, gth_out = get_check_data()
    act_func, grad_func = get_funcs(params)
    check(params, act_func, grad_func, input_data, gth_out)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--function', dest='function', type=str, default='sigmoid10',
                        help='identity/tanh/relu/sigmoid/sigmoid10/softmax')

    args = parser.parse_args()
    params = vars(args)
    main(params)
