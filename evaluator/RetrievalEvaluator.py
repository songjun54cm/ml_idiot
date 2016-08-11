__author__ = 'SongJun-Dell'
import numpy as np
from BasicEvaluator import BasicEvaluator

class RetrievalEvaluator(BasicEvaluator):
    def __init__(self):
        super(RetrievalEvaluator, self).__init__()

    def median_rank(self, ranks):
        sorted_ranks = np.sort(ranks)
        return sorted_ranks[len(sorted_ranks)/2]

    def mean_rank(self, ranks):
        rs = np.array(ranks, dtype=np.float)
        mean_rank = np.mean(rs)
        # print 'mean rank: %f' % mean_rank
        return mean_rank

    def mrr(self, ranks):
        rs = np.array(ranks, dtype=np.float)
        mrr = np.mean( 1.0 / rs )
        # print 'mrr: %f' % mrr
        return mrr

    def recall_at_ks(self, ranks, ks):
        """
        get recall @ Ks
        :param ranks: the ranks of correct index
        :param ks: values of K
        :return: print the result
        """
        if type(ks) != list:
            ks = [ks]
        r_at_ks = np.zeros(len(ks))
        rs = np.array(ranks, np.float)
        for i,k in enumerate(ks):
            r_at_ks[i] = self.recall_at_k(rs, k)
        return r_at_ks

    def recall_at_k(self, ranks, k):
        num = len(ranks)
        res = len(np.where(ranks<=k)[0])
        rk = res*1.0/num
        # print 'recall@%d: %f' % (k, rk)
        return rk