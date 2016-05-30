# ngram_evaluator.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from ngram_model import NgramModel
from nltk.util import ngrams
from math import e

class NgramEvaluator(object):
    '''
    Evaluates the likelihood of a given list of words appearing in text based on an N-gram
    language model.
    '''

    def __init__(self, n, alpha = None, ngram_model = None):
        self.n = n
        if ngram_model != None:
            self.language_model = ngram_model
        elif alpha != None:
            self.language_model = NgramModel(n, alpha = alpha)
        else:
            self.language_model = NgramModel(n)

    def score(self, phrase):
        '''
        Returns a score for the given phrase (a list of strings) based on the log likelihood of 
        seeing this phrase from the language model.
        '''
        # score the first n - 1 words in the phrase
        score = 0
        for m in range(min(self.n, len(phrase))):
            context = tuple(phrase[0:m])
            word = phrase[m]
            score += self.language_model.prob(context, word)

        # if the input phrase has fewer than n words, simply score the phrase without breaking
        # into n-grams
        if len(phrase) < self.n:
            return score

        # otherwise, sum the log probabilities of each n-gram
        phrase = [word.lower() for word in phrase]
        grams = ngrams(phrase, self.n)
        for gram in grams:
            context = gram[0:self.n - 1]
            word = gram[-1]
            score += self.language_model.prob(context, word)
        return score

    def perplexity(self, phrase):
        '''
        Returns the perplexity of the given phrase (a list of strings).
        '''
        log_prob = self.score(phrase)
        prob = pow(e, log_prob)
        # if the log_prob is low enough, prob will be 0.0, which causes a ZeroDivisionError
        if prob == 0:
            return float('inf')
        perplexity = pow(prob, -1 / len(phrase))
        return perplexity
