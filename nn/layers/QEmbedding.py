__author__ = 'SongJun-Dell'
import numpy as np
from BasicLayer import BasicLayer, micro_activate, get_action_function, get_gradient_function

class QEmbedding(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(QEmbedding, self).__init__(state, rng)

    def create_variables(self):
        if self.state['use_user']:
            self.cuid_emb_matrix, self.Wcuid_name = self.add_params((self.state['cuid_num'], self.state['cuid_emb_size']), 'Wcuid')
            self.regularize_param_names.append(self.Wcuid_name)
        if self.state['use_location']:
            self.city_emb_matrix, self.Wcity_name = self.add_params((self.state['city_num'], self.state['city_emb_size']), 'Wcity')
            self.regularize_param_names.append(self.Wcity_name)

        self.query_emb_matrix, self.Wquery_name = self.add_params((self.state['query_num'], self.state['query_emb_size']), 'Wquery')
        self.regularize_param_names.append(self.Wquery_name)

    # def init_grads(self):
    #     grad_params = dict()
    #     self.add_grad_param(grad_params, np.zeros(self.query_emb_matrix.shape), 'Wquery')
    #     if self.state['use_user']:
    #         self.add_grad_param(grad_params, np.zeros(self.cuid_emb_matrix.shape), 'Wcuid')
    #     if self.state['use_location']:
    #         self.add_grad_param(grad_params, np.zeros(self.city_emb_matrix.shape), 'Wcity')
    #
    #     return grad_params

    def activate(self, cuid_idx, city_idx, query_idx):
        res_list = [np.zeros((len(cuid_idx),1)), np.zeros((len(city_idx),1)), self.query_emb_matrix[query_idx, :]]
        if self.state['use_user']:
            res_list[0] = self.cuid_emb_matrix[cuid_idx, :]
        if self.state['use_location']:
            res_list[1] = self.city_emb_matrix[city_idx, :]
        return res_list

    def backward(self, grad_params, query_idxs, cuid_idxs, city_idxs, grad_query_embs, grad_cuid_embs, grad_city_embs):
        if len(grad_query_embs.shape) < 2: raise StandardError('grad_query_embs shape error')
        for i in xrange(len(query_idxs)):
            grad_params[self.Wquery_name][query_idxs[i], :] += grad_query_embs[i,:]
        if self.state['use_user'] and grad_cuid_embs is not None:
            for i in xrange(len(cuid_idxs)):
                grad_params[self.Wcuid_name][cuid_idxs[i], :] += grad_cuid_embs[i,:]
        if self.state['use_location'] and grad_city_embs is not None:
            for i in xrange(len(city_idxs)):
                grad_params[self.Wcity_name][city_idxs[i], :] += grad_city_embs[i,:]

    @staticmethod
    def fullfill_state(state):
        layer_state = {
            'layer_name': 'qembedding',
            'use_user': state['use_user'],
            'use_location': state['use_location'],
            'cuid_num': state['cuid_num'],
            'city_num': state['city_num'],
            'query_num': state['query_num'],
            'cuid_emb_size': state['cuid_emb_size'],
            'city_emb_size': state['city_emb_size'],
            'query_emb_size': state['query_emb_size']
        }
        return layer_state
