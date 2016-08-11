__author__ = 'SongJun-Dell'
from BasicEvaluator import BasicEvaluator

def unique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def intersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))

class NLPEvaluator(BasicEvaluator):
    def __init__(self):
        super(NLPEvaluator, self).__init__()

    def mean_rouge_bleu_at_ks(self, candidate_sentences, reference_sentences, ks):
        """
        :param candidate_sentences: [ sample_candi_sents, ... ]
        :param reference_sentences: [ sample_refer_sents, ... ]
        sample_candi_sents = [ sentence, ... ]
        sample_refer_sents = [ sentence, ... ]
        :param ks:
        :return:
        """
        mean_rouge_at_ks = list()
        mean_bleu_at_ks = list()
        rouge_list = list()
        bleu_list = list()
        for can_sents, ref_sents in zip(candidate_sentences, reference_sentences):
            rouges, bleus = self.get_rouge_bleu_one_sample(can_sents, ref_sents, ks)
            rouge_list.append(rouges)
            bleu_list.append(bleus)
        for i in xrange(len(ks)):
            mean_r_k = (sum([r[i] for r in rouge_list]) * 1.0) / (len(rouge_list)*1.0)
            mean_b_k = (sum([b[i] for b in bleu_list]) * 1.0) / (len(bleu_list)*1.0)
            mean_rouge_at_ks.append(mean_r_k)
            mean_bleu_at_ks.append(mean_b_k)
        return mean_rouge_at_ks, mean_bleu_at_ks

    def get_rouge_bleu_one_sample(self, cand_sentences, refe_sentences, ks):
        """
        :param cand_sentences: [ sentence, ... ]
        :param refe_sentences: [ sentence, ... ]
        :return:
        """
        rouges = list()
        bleus = list()
        for k in ks:
            candi_ngrams = self.get_ngrams(cand_sentences, k)
            refer_ngrams = self.get_ngrams(refe_sentences, k)
            inte = intersect(candi_ngrams, refer_ngrams)
            rouge = (len(inte)*1.0) / (len(refer_ngrams)*1.0)
            bleu = (len(inte)*1.0) / (len(candi_ngrams)*1.0)
            rouges.append(rouge)
            bleus.append(bleu)
        return rouges, bleus

    def get_ngrams(self, sentences, n):
        """
        :param sentences: [ sentence, ... ]
        :return: [ ngram, ... ]
        """
        ngrams = list()
        for sent in sentences:
            for i in xrange(len(sent)-n+1):
                ngrams.append('_'.join(sent[i:i+n]))
        ngrams = unique(ngrams)
        return ngrams
