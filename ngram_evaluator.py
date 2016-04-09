# ngram_evaluator.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from ngram_model import NgramModel
from nltk.util import ngrams

class NgramEvaluator(object):
    '''
    Evaluates the likelihood of a given list of words appearing in text based on an N-gram
    language model.
    '''

    def __init__(self, n):
        self.n = n
        self.language_model = NgramModel(n)

    def score(self, phrase):
        '''
        Returns a score for the given phrase (a list of strings) based on the log likelihood of 
        seeing this phrase from the language model.
        '''
        # if the input phrase has fewer than n words, add empty strings to create an n-tuple
        # to pass into the language model
        if len(phrase) < self.n:
            context = tuple([''] * (self.n - len(phrase)) + phrase[:-1])
            return self.language_model.prob(context, phrase[-1])

        phrase = [word.lower() for word in phrase]
        grams = ngrams(phrase, self.n)
        score = 0
        for gram in grams:
            context = gram[0:self.n - 1]
            word = gram[-1]
            score += self.language_model.prob(context, word)
        return score
