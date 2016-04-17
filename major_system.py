# major_system.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer
from number_encoder import (NumberEncoder, RandomGreedyEncoder, UnigramGreedyEncoder,
                            NgramContextEncoder)
from ngram_evaluator import NgramEvaluator

def main():
    '''
    Main function for Major System.
    '''

    # Demonstrate Pronouncer
    # print('=== Demonstrating the Pronouncer class ===')
    pronouncer = Pronouncer()
    # number_encoder = NumberEncoder(pronouncer = pronouncer)
    # words = ['apple', 'ABUSE', 'Rainbow']
    # for word in words:
    #     print('-- Pronunciation and decoding for \'{0}\' --'.format(word))
    #     print('\'{0}\' has pronunciations (with stress): {1}'.format(word,
    #         str(pronouncer.pronounce(word, strip_stress = False))))
    #     print('\'{0}\' has pronunciations (no stress): {1}'.format(word,
    #         str(pronouncer.pronounce(word))))
    #     print('\'{0}\' can be pronounced: {1}'.format(word,
    #         str(' '.join(pronouncer.pronounce(word)[0]))))
    #     print('\'{0}\' is an encoding for the number {1}'.format(word,
    #         number_encoder.decode_word(word)))
    # print('The sentence \'{0}\' is an encoding for the number {1}'.format(' '.join(words),
    #     number_encoder.decode_words(words)))

    # Demonstrate NgramEvaluator
    print('\n=== Demonstrating the NgramEvaluator class ===')
    evaluator = NgramEvaluator(2)
    sentences = ['this is a simple sentence', 'simple this a sentence is']
    for sentence in sentences:
        score = evaluator.score(sentence.split())
        print('The score for the sentence \'{0}\' is {1}.'.format(sentence, score))

    # Demonstrate RandomGreedyEncoder
    print('\n=== Demonstrating the RandomGreedyEncoder class ===')
    vocab_sizes = [1000, 10000, 50000, None]
    numbers = ['123', '123', '451', '451', '12345', '0123456789',
               '31415926535897932384626433832795028841971693993751']
    for vocab_size in vocab_sizes:
        print('-- Restricting vocabulary size to {0} --'.format(vocab_size))
        random_greedy_encoder = RandomGreedyEncoder(pronouncer = pronouncer, max_word_length = 2,
            max_vocab_size = 50000)
        for max_word_length in [1, 2, 3, 10]:
            print('-- Encoding with max_word_length {0} --'.format(max_word_length))
            for number in numbers:
                encoding = random_greedy_encoder.encode_number(number, 
                    max_word_length=max_word_length)
                decoding = random_greedy_encoder.decode_words(encoding)
                print('The number \'{0}\' can be encoded as \'{1}\' (which decodes to \'{2}\').'.
                    format(number, encoding, decoding))
                score = evaluator.score(encoding)
                print('The score for this encoding is {0}.'.format(score))

    # Demonstrate UnigramGreedyEncoder
    print('\n=== Demonstrating the UnigramGreedyEncoder class ===')
    unigram_greedy_encoder = UnigramGreedyEncoder(pronouncer = pronouncer, max_word_length = 2)
    numbers = ['123', '451', '12345', '0123456789',
               '31415926535897932384626433832795028841971693993751']
    for max_word_length in [1, 2, 3, 10]:
        print('-- Encoding with max_word_length {0} --'.format(max_word_length))
        for number in numbers:
            encoding = unigram_greedy_encoder.encode_number(number, max_word_length=max_word_length)
            decoding = unigram_greedy_encoder.decode_words(encoding)
            print('The number \'{0}\' can be encoded as \'{1}\' (which decodes to \'{2}\').'.
                format(number, encoding, decoding))
            score = evaluator.score(encoding)
            print('The score for this encoding is {0}.'.format(score))

    # Demonstrate NgramContextEncoder
    print('\n=== Demonstrating the NgramContextEncoder class ===')
    for ngram_n in [1, 2, 3]:
        print('\n-- Encoding with n-gram model n = {0} --'.format(ngram_n))
        ngram_context_encoder = NgramContextEncoder(pronouncer = pronouncer, max_word_length = 5,
            n = ngram_n, alpha = 0.1, select_most_likely = True)
        evaluator = NgramEvaluator(ngram_n)
        numbers = ['123', '451', '12345', '0123456789',
                   '31415926535897932384626433832795028841971693993751']
        for number in numbers:
            encoding = ngram_context_encoder.encode_number(number)
            decoding = ngram_context_encoder.decode_words(encoding)
            print('The number \'{0}\' can be encoded as \'{1}\' (which decodes to \'{2}\').'.
                format(number, encoding, decoding))
            score = evaluator.score(encoding)
            print('The score for this encoding is {0}.'.format(score))

if __name__ == "__main__":
    main()
