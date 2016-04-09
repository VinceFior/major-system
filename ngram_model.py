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
        self.n = n 

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
        Returns the log probability of the word given the context. The context should be a tuple 
        with no more than n - 1 elements.
        '''
        context = tuple([token.lower() for token in context])
        word = word.lower()
        for index, gram in enumerate(self.grams[-len(context) - 1:]):
            # truncate context as gram size decreases
            this_context = context[index:]
            if gram[this_context][word] != 0:
                prob = gram[this_context][word] / gram[this_context].N()
                return log(prob * pow(self.alpha, index))
        return 0
