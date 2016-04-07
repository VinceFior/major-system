# major_system.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer
from number_encoder import NumberEncoder
from number_encoder import RandomGreedyEncoder

def main():
    '''
    Main function for Major System.
    '''

    # Demonstrate Pronouncer
    print('=== Demonstrating the Pronouncer class ===')
    pronouncer = Pronouncer()
    number_encoder = NumberEncoder(pronouncer = pronouncer)
    words = ['apple', 'ABUSE', 'Rainbow']
    for word in words:
        print('-- Pronunciation and decoding for \'{0}\' --'.format(word))
        print('\'{0}\' has pronunciations (with stress): {1}'.format(word,
            str(pronouncer.pronounce(word, strip_stress = False))))
        print('\'{0}\' has pronunciations (no stress): {1}'.format(word,
            str(pronouncer.pronounce(word))))
        print('\'{0}\' can be pronounced: {1}'.format(word,
            str(' '.join(pronouncer.pronounce(word)[0]))))
        print('\'{0}\' is an encoding for the number {1}'.format(word,
            number_encoder.decode_word(word)))
    print('The sentence \'{0}\' is an encoding for the number {1}'.format(' '.join(words),
        number_encoder.decode_words(words)))

    # Demonstrate RandomGreedyEncoder
    print('\n=== Demonstrating the RandomGreedyEncoder class ===')
    random_greedy_encoder = RandomGreedyEncoder(pronouncer = pronouncer, max_word_length = 2)
    numbers = ['123', '123', '451', '451', '12345', '0123456789']
    for max_word_length in [1, 2, 3, 10]:
        print('-- Encoding with max_word_length {0} --'.format(max_word_length))
        for number in numbers:
            encoding = random_greedy_encoder.encode_number(number, max_word_length=max_word_length)
            decoding = random_greedy_encoder.decode_words(encoding.split(' '))
            print('The number \'{0}\' can be encoded as \'{1}\' (which decodes to \'{2}\').'.
                format(number,encoding, decoding))

if __name__ == "__main__":
    main()
