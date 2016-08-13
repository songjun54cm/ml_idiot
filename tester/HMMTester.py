__author__ = 'SongJun-Dell'


class HMMTester(object):
    def __init__(self):
        pass

    def test(self, model, test_data, id_to_fea):
        obs_seqs = test_data['obs_seqs']
        state_seqs = test_data['state_seqs']

        for si in xrange(obs_seqs):
            obss = obs_seqs[si]
            states = state_seqs[si]
            pred_states = model.predict_states(obss, states)
