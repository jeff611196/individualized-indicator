# In[]
import numpy as np
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import matutils
from six import string_types


class cosine_similar():
    def __init__(self, embeddings):
        self.sentences_keys = list(embeddings.keys())
        self.embeddings = embeddings

    def most_similar(self, positive=[], negative=[], topn=10, restrict_vocab=None, indexer=None):
        self.init_sims()
        if isinstance(positive, string_types) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]
        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [
            (word, 1.0) if isinstance(word, string_types + (ndarray,)) else word
            for word in positive
        ]
        negative = [
                (word, -1.0) if isinstance(word, string_types + (ndarray,)) else word
                for word in negative
        ]
        
        # compute the weighted average of all words
        all_words, mean = set(), []
        for word, weight in positive + negative:
            if isinstance(word, ndarray):
                mean.append(weight * word)
            elif word in self.sentences_keys:
                mean.append(weight * self.syn0norm[self.sentences_keys.index(word)])
                all_words.add(self.sentences_keys.index(word))
            else:
                raise KeyError("word '%s' not in vocabulary" % word)

        if not mean:
            raise ValueError("cannot compute similarity with no input")
        mean = matutils.unitvec(array(mean).mean(axis=0)).astype(REAL)

        if indexer is not None:
            return indexer.most_similar(mean, topn)

        limited = self.syn0norm if restrict_vocab is None else self.syn0norm[:restrict_vocab]
        dists = dot(limited, mean)
        if not topn:
            return dists
        best = matutils.argsort(dists, topn=topn + len(all_words), reverse=True)
        # ignore (don't return) words from the input
        result = dict([(self.sentences_keys[sim], float(dists[sim])) for sim in best if sim not in all_words])
        #return result[:topn]
        return result

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        '''if getattr(self, 'syn0norm', None) is None or replace:
            #logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in xrange(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)'''
        self.syn0norm = list(range(len(self.embeddings)))
        for idx, node in enumerate(self.embeddings):
            self.syn0norm[idx] = self.embeddings[node] / sqrt((self.embeddings[node] ** 2).sum(-1))
# %%
