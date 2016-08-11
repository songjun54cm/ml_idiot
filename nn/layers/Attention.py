__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function, softmax_func, get_grad_softmax

class Attention(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(Attention, self).__init__(state, rng)

    def create_variables(self):
        # e_i = w_e * tanh(Wx * x_i + Wh * h + b_a)
        self.Wx, self.Wx_name = self.add_params((self.state['x_size'], self.state['att_size']), 'Wx')
        self.regularize_param_names.append(self.Wx_name)
        self.Wh, self.Wh_name = self.add_params((self.state['h_size'], self.state['att_size']), 'Wh')
        self.regularize_param_names.append(self.Wh_name)
        self.ba, self.ba_name = self.add_params((1, self.state['att_size']), 'ba')
        self.wa, self.wa_name = self.add_params((1, self.state['att_size']), 'wa')

        self.score_act = get_action_function('tanh')
        self.score_grad_act = get_gradient_function('tanh')

    def activate(self, input_x, input_h):
        """
        :param input_x: (N, x_size) matrix
        :param input_h: (1, h_size) vector
        :return: (1, x_size) vector, the attention vector of current step
        """
        score_vec = self.score_act(input_x.dot(self.Wx) + input_h.dot(self.Wh) + self.ba) # (N, att_size)
        # (N, 1)
        vs = score_vec.dot(self.wa.transpose())
        # attend weight
        # (N, 1)
        ve = softmax_func(vs.transpose()).transpose()
        z = ve.transpose().dot(input_x)

        cache = {
            'input_x': input_x,
            'input_h': input_h,
            'att_weight': ve,
            'score_vec': score_vec,
        }

        return z, cache

    def backward_step(self, grad_params, grad_attention_out, attend_cache):
        input_x = attend_cache['input_x']
        input_h = attend_cache['input_h']
        ve = attend_cache['att_weight']
        score_vec = attend_cache['score_vec']
        # grad_ve = grad_attention_out.dot(input_x.transpose()) # (1, N)
        # (N, 1)
        grad_ve = input_x.dot(grad_attention_out.transpose())
        # (N, 1)
        grad_vs = get_grad_softmax(ve.transpose(), grad_ve.transpose()).transpose()

        grad_params[self.wa_name] += grad_vs.transpose().dot(score_vec)
        grad_unact_score_vec = (grad_vs.dot(self.wa)) * self.score_grad_act(score_vec)

        grad_params[self.ba_name] += np.sum(grad_unact_score_vec, axis=0, keepdims=True)

        grad_params[self.Wh_name] += input_h.transpose().dot(np.sum(grad_unact_score_vec,axis=0,keepdims=True))
        grad_h = np.sum(grad_unact_score_vec, axis=0, keepdims=True).dot(self.Wh.transpose())

        grad_params[self.Wx_name] += input_x.transpose().dot(grad_unact_score_vec)

        grad_input_x = ve.dot(grad_attention_out) + grad_unact_score_vec.dot(self.Wx.transpose())

        return grad_input_x, grad_h

    @staticmethod
    # gradient descent check data
    def gdc_data():
        x_size = 2
        att_size = 3
        h_size = 2
        x_num = 3
        layer_state = {
            'layer_name': 'attend',
            'x_size': x_size,
            'att_size': att_size,
            'h_size': h_size
        }
        input_x = np.random.RandomState(1).rand(x_num, x_size)
        input_h = np.random.RandomState(2).rand(1, h_size)
        ground_truth_out = np.random.RandomState(3).rand(1, x_size)

        gdc_data = {
            'layer_state': layer_state,
            'input_data': {'input_x': input_x, 'input_h': input_h},
            'gth_out': ground_truth_out
        }
        return gdc_data

    def gdc_activate(self, input_data):
        input_x = input_data['input_x']
        input_h = input_data['input_h']

        layer_out, cache = self.activate(input_x, input_h)

        return layer_out, cache

    def gdc_backward(self, grad_out, cache):
        grad_params = dict()
        for p in self.params.keys():
            grad_params[p] = np.zeros(self.params[p].shape)
        self.backward_step(grad_params, grad_out, cache)

        return grad_params
