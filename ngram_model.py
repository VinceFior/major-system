# ngram_model.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist
from nltk.util import ngrams
from math import log

class NgramModel(object):
    '''
    N-gram language model with stupid backoff.
    '''
    def __init__(self, n, alpha = 0.4, brown_categories = None):
        '''
        Initializes NgramModel with a list of conditional frequency distributions representing
        N-grams, (N-1)-grams, ...., bigrams, unigrams from the Brown corpus.
        '''
        if brown_categories == None:
            brown_categories = brown.categories()
        samples = [[]] * n
        for category in brown_categories:
            text = brown.words(categories=category)
            text = [word.lower() for word in list(text)]
            for index, m in enumerate(range(n, 0, -1)):
                igrams = ngrams(text, m)
                igrams = [(igram[0:m - 1], igram[-1]) for igram in list(igrams)]
                samples[index] += igrams
        # list of N-grams with descending values of N
        self.grams = []
        for sample in samples:
            self.grams += [ConditionalFreqDist(sample)]

        # multiplier for each level of backoff
        self.alpha = alpha

    def prob(self, context, word):
        '''
        Returns the log probability of the word given the context, which is expected to be a tuple
        of strings (empty in the case of a unigram).
        '''
        context = tuple([token.lower() for token in context])
        word = word.lower()
        for index, gram in enumerate(self.grams):
            if gram[context][word] != 0:
                return log(gram[context][word]) * pow(self.alpha, index)
        return 0
