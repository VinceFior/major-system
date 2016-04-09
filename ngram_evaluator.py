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
        # score the first n - 1 words in the phrase
        score = 0
        for m in range(min(self.n, len(phrase))):
            context = tuple(phrase[0:m])
            word = phrase[m]
            score += self.language_model.prob(context, word)

        # if the input phrase has fewer than n words, simply score the phrase without breaking
        # into n-grams
        if len(phrase) < self.n:
            return score #+ self.language_model.prob(tuple(phrase[0:len(phrase) - 1]), phrase[-1])

        # otherwise, sum the log probabilities of each n-gram
        phrase = [word.lower() for word in phrase]
        grams = ngrams(phrase, self.n)
        for gram in grams:
            context = gram[0:self.n - 1]
            word = gram[-1]
            score += self.language_model.prob(context, word)
        return score
