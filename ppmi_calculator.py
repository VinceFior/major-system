# ngram_model.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from ngram_model import NgramModel
from math import exp, log

class PPMICalculator(object):
    '''
    PPMICalculator provides the function ppmi() to calculate the positive pointwise mutual
    information.
    '''

    def __init__(self, alpha = 0.1):
        self.unigram = NgramModel(n = 1, alpha = alpha)
        self.bigram = NgramModel(n = 2, alpha = alpha)

    def ppmi(self, x, y):
        '''
        Returns the positive pointwise mutual information, which is calculated by as the max of 0
        and the log of (P(X, Y) / (P(X) * P(Y))).
        '''
        num = exp(self.bigram.prob((x,), y))
        den = exp(self.unigram.prob((), x)) * exp(self.unigram.prob((), y))
        pmi = log(num / den)
        return max(0, pmi)
