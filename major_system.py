# major_system.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer
from number_encoder import NumberEncoder

def main():
    '''
    Main function for Major System.
    '''
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

if __name__ == "__main__":
    main()