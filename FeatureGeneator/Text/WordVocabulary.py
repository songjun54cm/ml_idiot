__author__ = 'JunSong<songjun54cm@gmail.com>'
import argparse
import time
def get_word_vocabulary_from_docs(docs,
                                  low_thresh_ratio=0.0,
                                  high_thresh_ratio=0.0,
                                  doc_count_low_thresh=0,
                                  with_all_voca=False,
                                  with_bow=False,
                                  verbose=False):
    '''
    calculate the word dictionary for list of documents
    :param docs: [ [words], [words], ... ], list of list of word
    :param low_thresh_ratio: drop out low frequency ratio
    :param high_thresh_ratio: drop out high frequency ratio
    :param with_bow: give bag-of-words feature (word frequency(TF)) at the same time
    :return:
    word_voca:{
        word2id: { word:{'id':x, 'freq':x, 'doc_freq':x}, ... },
        id2word: [ word, ... ]
    }
    '''
    word_dics = {}
    doc_account = len(docs) * 1.0
    word_account = sum([len(d) for d in docs]) * 1.0
    doc_words = list()
    n = 0
    start_time = time.time()
    for doc in docs:
        word_dic_doc = dict()
        for word in doc:
            word_dic_doc[word] = word_dic_doc.get(word, 0.0) + 1.0
        for w,c in word_dic_doc.iteritems():
            if w not in word_dics:
                word_dics[w] = {'freq': c, 'doc_freq': 1.0}
            else:
                word_dics[w]['freq'] += c
                word_dics[w]['doc_freq'] += 1.0

        if with_bow:
            doc_len = len(doc)*1.0
            for w in word_dic_doc.keys():
                word_dic_doc[w] /= doc_len
            doc_words.append(word_dic_doc)

        n += 1
        if n%1000 == 0:
            if verbose:
                print('%d documents processed. in %f seconds' % (n, time.time()-start_time))
            start_time = time.time()

    sorted_words = sorted(word_dics.iteritems(), key=lambda x: x[1]['freq'], reverse=True)
    start_pos = int(len(sorted_words) * low_thresh_ratio)
    end_pos = int(len(sorted_words) * (1.0 - high_thresh_ratio))
    use_words = sorted_words[start_pos:end_pos]
    use_words2 = list()
    for word_info in use_words:
        if word_info[1]['doc_freq'] >= doc_count_low_thresh:
            use_words2.append(word_info)
    use_words = [(x[0], {'id': i, 'freq':x[1]['freq']/word_account, 'doc_freq':x[1]['doc_freq']/doc_account})
                 for i,x in enumerate(use_words2)]
    id2word = [x[0] for x in use_words]
    word2id = dict(use_words)
    print('get word vocabulary with size %d' % len(id2word))
    word_voca = {
        'word2id': word2id,
        'id2word': id2word
    }
    print('vocabulary contains %d words, use %d words.' % (len(sorted_words), len(id2word)))
    res = {
        'word_voca': word_voca,
    }

    if with_all_voca:
        # [(word, {occ,doc_occ}), ...]
        res['all_voca'] = sorted_words

    if with_bow:
        import numpy as np
        from scipy.sparse import lil_matrix
        doc_bow_feats = lil_matrix((len(docs), len(id2word)),  dtype=np.float32)
        print('documents bow of words feature with shape: %s' % str(doc_bow_feats.shape))
        for i, dws in enumerate(doc_words):
            for w,f in dws.iteritems():
                winfo = word2id.get(w, False)
                if winfo:
                    doc_bow_feats[i,winfo['id']] = f
        res['doc_bow_feats'] = doc_bow_feats
    return res

def main(state):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    state = vars(args)
    main(state)