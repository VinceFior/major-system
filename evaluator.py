# evaluator.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from ngram_model import NgramModel
from nltk import bigrams

class NgramEvaluator(object):
    '''
    Evaluates the likelihood of a given list of words appearing in text based on an N-gram
    language model.
    '''

    def __init__(self, n):
        self.language_model = NgramModel(n)

    def score(self, phrase):
        grams = bigrams(phrase)
        score = 1
        for context, word in grams:
            score += self.language_model.prob((context,), word)
        return score

# def main():
#     e = NgramEvaluator(2)
#     first = ['Grand', 'Jury', 'said']
#     first = [word.lower() for word in first]
#     sentence = ['this', 'is', 'a', 'simple', 'sentence']
#     mash = ['simple', 'this', 'a', 'sentence', 'is']
#     gibberish = ['adfasd', 'eiua', 'ghrvl', 'iuy', 'bfx']
#     print(e.score(first))
#     print(e.score(sentence))
#     print(e.score(mash))
#     print(e.score(gibberish))

# if __name__ == "__main__":
#     main()