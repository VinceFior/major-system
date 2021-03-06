# ngram_model.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, FreqDist, MLEProbDist, ConditionalProbDist
from nltk.util import ngrams
from math import log

class NgramBase(object):
    '''
    NgramBase is the base class for any N-gram language model.
    '''

    def __init__(self, n, alpha = 0.1, brown_categories = None):
        '''
        Initializes NgramBase with a list of conditional frequency distributions representing
        N-grams, (N-1)-grams, ...., bigrams, unigrams from the Brown corpus.
        '''
        self.n = n 

        if brown_categories == None:
            brown_categories = brown.categories()
        samples = [[]] * n
        sents = self._get_sentences(brown_categories)
        for sent in sents:
            sent = [word.lower() for word in sent]
            if sent[-1].isalpha():
                sent += ['.']
            for index, m in enumerate(range(n, 0, -1)):
                igrams = ngrams(sent, m)
                igrams = [(igram[0:m - 1], igram[-1]) for igram in list(igrams)]
                samples[index] += igrams
        # list of N-grams with descending values of N
        self.grams = []
        for sample in samples:
            self.grams += [ConditionalFreqDist(sample)]

        # multiplier for each level of backoff
        self.alpha = alpha

    def _get_sentences(self, brown_categories):
        '''
        Returns a list of lists of strings representing sentences.
        '''
        raise NotImplementedError()

    def prob(self, context, element):
        '''
        Returns the log probability of the element given the context. The context should be a tuple 
        with no more than n - 1 elements.
        '''
        if self.alpha == 0:
            count = self.grams[0][context][element]
            if count != 0:
                prob = count / self.grams[0][context].N()
                return log(prob)
            else:
                return -float('inf')

        context = tuple([token.lower() for token in context])
        element = element.lower()
        for index, gram in enumerate(self.grams[-len(context) - 1:]):
            # truncate context as gram size decreases
            this_context = context[index:]
            if gram[this_context][element] != 0:
                prob = gram[this_context][element] / gram[this_context].N()
                return log(prob * pow(self.alpha, index))
        return -float('inf')


class NgramModel(NgramBase):
    '''
    N-gram language model with stupid backoff.
    '''

    def _get_sentences(self, brown_categories):
        '''
        Returns a list of lists of strings representing the words in each sentence.
        '''
        return list(brown.sents(categories=brown_categories))


class NgramPOSModel(NgramBase):
    '''
    N-gram part-of-speech language model with stupid backoff.
    '''
    
    def _get_sentences(self, brown_categories):
        '''
        Returns a list of lists of strings representing the part-of-speech of each word in each
        sentence.
        '''
        tagged_sents = list(brown.tagged_sents(categories=brown_categories))
        tags = []
        for sent in tagged_sents:
            tags += [[tag for word, tag in sent]]
        return tags
