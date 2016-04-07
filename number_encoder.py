# number_encoder.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer

class NumberEncoder(object):
    '''
    NumberEncoder is the base class for any model that encodes a number (string of digits) as a
    series of words.
    If given a pronouncer, uses it. Also, if given a dictionary mapping from phonemes to digits
    (length-one strings), uses it.
    '''
    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None):
        self.pronouncer = pronouncer
        if phoneme_to_digit_dict:
            self.phoneme_to_digit_dict = phoneme_to_digit_dict
        else:
            self.phoneme_to_digit_dict = self._get_phoneme_to_digit_dict()

    def encode_number(self, number):
        '''
        Encodes the given number (string of digits) as a series of words.
        '''
        raise NotImplementedError()

    def decode_words(self, words):
        '''
        Decodes the given list of words as a number (string of digits).
        '''
        return ''.join([self.decode_word(word) for word in words])

    def decode_word(self, word):
        '''
        Decodes the given word to a list of digits (length-one strings).
        '''
        phonemes_list = self.pronouncer.pronounce(word)
        # in the case of multiple pronunciations, check each one's decoding
        possible_decodings = []
        for phonemes in phonemes_list:
            decoding = ''
            for phoneme in phonemes:
                # some phonemes do not map to a digit, so we exclude those phonemes
                if phoneme in self.phoneme_to_digit_dict:
                    decoding += self.phoneme_to_digit_dict[phoneme]
            possible_decodings += [decoding]
        # if all possible pronunciations yield the same decoding, that's fine; otherwise, we fail
        if not all(decoding == possible_decodings[0] for decoding in possible_decodings):
            print('Multiple possible decodings for word \'{0}\': {1}'.format(word, 
                possible_decodings))
            return None
        return possible_decodings[0]

    def _get_phoneme_to_digit_dict(self):
        '''
        Returns a dictionary mapping phonemes to a digit (length-one string).
        Note that not all phonemes map to a digit. These unused phonemes are not included.
        This implementation is written for the ARPAbet phoneme set using the most common mapping.
        '''
        # for convenience, we define the mapping from digit to phonemes first, then invert it
        # this mapping uses every non-vowel phoneme except 'HH' (aspirate) and 'NG' (nasal), but
        # it does use the "vowel" 'ER'
        digit_to_phonemes_dict = {
        '0': ('S', 'Z'),
        '1': ('T', 'D', 'TH', 'DH'),
        '2': ('N'),
        '3': ('M'),
        '4': ('R', 'ER'), # we treat 'ER' as if it were just 'R'
        '5': ('L'),
        '6': ('CH', 'JH', 'SH', 'ZH'),
        '7': ('K', 'G'),
        '8': ('F', 'V'),
        '9': ('P', 'B')
        }
        phoneme_to_digit_dict = dict()
        for digit, phonemes in digit_to_phonemes_dict.items():
            for phoneme in phonemes:
                phoneme_to_digit_dict[phoneme] = digit
        return phoneme_to_digit_dict
