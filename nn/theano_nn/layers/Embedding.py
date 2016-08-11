__author__ = 'SongJun-Dell'
from BasicLayer import BasicLayer, add_to_params, create_shared
import numpy as np

class Embedding(BasicLayer):
    def __init__(self, state, rng=np.random.RandomState(1234)):
        super(Embedding, self).__init__(state)
        self.create_variables()

    def create_variables(self):
        self.W = add_to_params(self.params, create_shared((self.state['voca_size'], self.state['emb_size']), rng=self.rng, name=self.get_variable_name('W')))
        self.regularize_params.append(self.W)

    def forwward(self, idxs):
        # idxs: (num_of_sample, max_len), list of lists
        # assert(idxs.ndim==2, 'ndim of idxs not equals to 2')
        embs = self.W[idxs, :]
        return embs

    @staticmethod
    def fullfill_state(state):
        layer_state = {
            'layer_name': state.get('layer_name', 'embedding'),
            'voca_size': state['voca_size'],
            'emb_size': state['emb_size']
        }
        return layer_state