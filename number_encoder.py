# number_encoder.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer
from random import sample # for RandomGreedyEncoder
from itertools import product # for RandomGreedyEncoder

class NumberEncoder(object):
    '''
    NumberEncoder is the base class for any model that encodes a number (string of digits) as a
    series of words.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None):
        '''
        Initializes the NumberEncoder. If given a pronouncer, uses it. Also, if given a dictionary
        mapping from phonemes to digits (length-one strings), uses it.
        '''
        self.pronouncer = pronouncer
        if phoneme_to_digit_dict:
            self.phoneme_to_digit_dict = phoneme_to_digit_dict
        else:
            self.phoneme_to_digit_dict = self._get_phoneme_to_digit_dict()

    def encode_number(self, number):
        '''
        Encodes the given number (string of digits) as a series of words.
        '''
        # this method is what really distinguishes one encoder from another
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
        possible_decodings = self._decode_phonemes_list(phonemes_list)
        # if all possible pronunciations yield the same decoding, that's fine; otherwise, we fail
        if not all(decoding == possible_decodings[0] for decoding in possible_decodings):
            print('Multiple possible decodings for word \'{0}\': {1}'.format(word,
                possible_decodings))
            return None
        return possible_decodings[0]

    def _decode_phonemes_list(self, phonemes_list):
        '''
        Decodes the given list of phoneme lists to a list of digits (length-one strings).
        '''
        possible_decodings = []
        for phonemes in phonemes_list:
            decoding = ''
            for phoneme in phonemes:
                # some phonemes do not map to a digit, so we exclude those phonemes
                if phoneme in self.phoneme_to_digit_dict:
                    decoding += self.phoneme_to_digit_dict[phoneme]
            possible_decodings += [decoding]
        return possible_decodings

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


class GreedyEncoder(NumberEncoder):
    '''
    GreedyEncoder is the base class for any encoder that greedily tries to make as long words
    as possible (within an optional limit) and somehow picks from the set of longest words.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = 2):
        '''
        Initializes the RandomGreedyEncoder. The encoder will greedily group digits to make words
        as long as possible (where the length of a word refers to the number of digits it encodes).
        '''
        super(GreedyEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict)
        # with the default encoding, the maximum number of digits in any word is 19
        max_digits_per_word = 19
        # if we were given an unusual max_word_length, set it to max_digits_per_word
        if max_word_length == None or max_word_length < 1 or max_word_length > max_digits_per_word:
            max_word_length = max_digits_per_word
        self.max_word_length = max_word_length
        # to aid encode_number(), we set up a mapping from phoneme sequences to words
        self.phonemes_to_words_dict = self._get_phonemes_to_words_dict()

    def select_encoding(self, encodings):
        '''
        Selects one encoding from the given list of possible encodings.
        '''
        # this method is what really distinguishes one greedy encoder from another
        raise NotImplementedError()

    def encode_number(self, number, max_word_length = None):
        '''
        Encodes the given number (string of digits) as a series of words. Greedily groups digits
        in chunks as long as possible, up to max_word_length (if given, otherwise
        self.max_word_length). Once a length is found that produces at least one word, the encoder
        selects a word of that length via self.select_encoding().
        '''
        # if not given max_word_length, use class default
        if max_word_length == None:
            max_word_length = self.max_word_length
        encoded_index = 0 # the last index of number we've encoded, inclusive
        encodings = []
        while encoded_index < len(number):
            # find an encoding for this chunk
            has_found_encoding = False
            chunk_length = max_word_length
            while chunk_length > 0 and not has_found_encoding:
                number_chunk = number[encoded_index : encoded_index + chunk_length]
                chunk_encodings = self._encode_number_chunk(number_chunk)
                if len(chunk_encodings) > 0:
                    # select one encoding from chunk_encodings
                    chunk_encoding = self.select_encoding(chunk_encodings)
                    has_found_encoding = True
                    encodings += [chunk_encoding]
                    encoded_index += chunk_length
                else:
                    chunk_length -= 1
            if not has_found_encoding:
                number_chunk = number[encoded_index : encoded_index + max_word_length]
                print('Cannot find encoding for number chunk \'{0}\''.format(number_chunk))
                return None
        return ' '.join(encodings)

    def _encode_number_chunk(self, number_chunk):
        '''
        Helper method for encode_number() that returns a list of all single-word encodings for the
        given number_chunk.
        '''
        # enumerate all possible sequences of phonemes for number_chunk
        possible_phonemes = []
        for digit in number_chunk:
            possible_phonemes += [tuple([phoneme for phoneme in self.phoneme_to_digit_dict
                                         if digit == self.phoneme_to_digit_dict[phoneme]])]
        # check every possible phoneme sequence in self.phonemes_to_words_dict and add every word
        possible_encodings = []
        for phoneme_sequence in product(*possible_phonemes):
            if phoneme_sequence in self.phonemes_to_words_dict:
                possible_encodings += self.phonemes_to_words_dict[phoneme_sequence]
        return possible_encodings

    def _get_phonemes_to_words_dict(self):
        '''
        Returns a dictionary that maps every sequence (tuple) of phonemes (that are included in
        self.phoneme_to_digit_dict) to a list of all words in
        self.pronouncer.pronunciation_dictionary that can be pronounced as that sequence.
        Only includes keys with values (i.e., only includes sequences that actually appear).
        '''
        phonemes_to_words_dict = dict()
        for word, pronunciations in self.pronouncer.pronunciation_dictionary.items():
            pronunciations = self.pronouncer.strip_phonemes_stress(pronunciations)
            # if any of the pronunciations yield different decodings, do not include this word
            possible_decodings = self._decode_phonemes_list(pronunciations)
            if not all(decoding == possible_decodings[0] for decoding in possible_decodings):
                continue
            # include the phoneme sequence for every pronunciation
            for pronunciation in pronunciations:
                included_phonemes = tuple([phoneme for phoneme in pronunciation
                                           if phoneme in self.phoneme_to_digit_dict])
                # if we've already included a word for this phoneme sequence, extend its list
                if included_phonemes in phonemes_to_words_dict:
                    phonemes_to_words_dict[included_phonemes] += [word]
                else:
                    phonemes_to_words_dict[included_phonemes] = [word]
        return phonemes_to_words_dict

    def _get_max_digits_per_word(self):
        '''
        Searches through self.pronouncer.pronunciation_dictionary and returns the maximum number
        of digits that can be encoded in any word.
        '''
        # with the default phoneme_to_digit_dict and pronouncer.pronunciation_dictionary, returns
        # 19 for 'SUPERCALIFRAGILISTICEXPEALIDOSHUS'
        max_digits = 0
        for word, pronunciations in self.pronouncer.pronunciation_dictionary.items():
            pronunciations = self.pronouncer.strip_phonemes_stress(pronunciations)
            for pronunciation in pronunciations:
                digits_in_word = 0
                for phoneme in pronunciation:
                    if phoneme in self.phoneme_to_digit_dict:
                        digits_in_word += 1
                max_digits = max(digits_in_word, max_digits)
        return max_digits


class RandomGreedyEncoder(GreedyEncoder):
    '''
    RandomGreedyEncoder is a GreedyEncoder that selects which encoding to use randomly.
    '''

    def select_encoding(self, encodings):
        '''
        Randomly selects one encoding from the given encodings.
        '''
        return sample(encodings, 1)[0]
