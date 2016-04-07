# pronouncer.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.
# This class uses the CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

import codecs # for reading the CMU dictionary file

class Pronouncer(object):
    '''
    Pronouncer knows how to pronounce words. It will even tell you how a word is pronounced,
    if you ask nicely.
    '''
    def __init__(self, pro_dict_file = 'phonemes/cmudict-07b.txt', 
        phonemes_file = 'phonemes/cmudict-0-7b-phones.txt', files_encoding = 'latin-1',
        strip_stress = True):
        '''
        Initializes the Pronouncer. In particular, this method gets and parses the given
        pronunciation dictionary and phonemes file using the given text encoding.
        You can also set whether this Pronouncer should automatically strip out stress markers.
        '''
        self.pronunciation_dictionary = self._get_pronunciation_dict(pro_dict_file, files_encoding)
        self.phonemes_set = self._get_phonemes(phonemes_file, files_encoding)
        self.strip_stress = strip_stress

    def pronounce(self, word, strip_stress = None):
        '''
        Given a (known) word, returns a list of list of phonemes (strings) representing the 
        possible pronunciations of the word. These phonemes (optionally) include the stress markers
        on the phonemes (e.g., 'AH0' as opposed to just 'AH').
        '''
        # if not given strip_stress, use class default
        if strip_stress == None:
            strip_stress = self.strip_stress
        upper_word = word.upper() # our dictionary is all uppercase
        if not upper_word in self.pronunciation_dictionary:
            print('Do not know how to pronounce \'{0}\''.format(word))
            return None
        phonemes_list = self.pronunciation_dictionary[upper_word]
        if strip_stress:
            phonemes_list = self._strip_phonemes_stress(phonemes_list)
        return phonemes_list

    def _get_pronunciation_dict(self, dict_filename, encoding, words_to_include = None):
        ''' Returns a dict where each key is a string like 'WORD' and each value is a list of
            lists of phonemes (with stress) for the word, such as [['W', 'ER1', 'D']].
            Assumes the given dict_filename is a version of the CMU Pronouncing Dictionary.
            Only includes words in the given set of words_to_include (if given).
            Opens the dict_file with the given encoding.
        '''
        # we iterate through the given file and build up our dictionary
        pronunciation_dictionary = dict()
        with codecs.open(dict_filename, 'r', encoding) as f:
            # each line is of the form 'WORD S Y MB OLS' or 'WORD(1) S Y MB OLS'
            #   ex., 'ABUSES(1)  AH0 B Y UW1 Z IH0 Z'
            # note that 'WORD' might contain non-letter characters, like apostrophes
            for line in f:
                if line[:3] == ';;;': # comment lines start with ';;;'
                    continue
                line = line[:-1] # remove '\n' from end of line
                word_and_pron = line.split(' ', 1) # split line into word and pronunciation
                word = word_and_pron[0]
                pronunciation = word_and_pron[1][1:] # remove leading space
                # if the word is of the form 'WORD(1)', prune off the '(1)'
                if word[-1] == ')':
                    word = word[:-3]
                # exclude words not in words_to_include
                if not words_to_include or word in words_to_include:
                    # pronunciation is a string of the form 'S Y MB OLS'
                    phonemes = pronunciation.split(" ")
                    if word in pronunciation_dictionary:
                        pronunciation_dictionary[word] += [phonemes]
                    else:
                        pronunciation_dictionary[word] = [phonemes]
        return pronunciation_dictionary

    def _get_phonemes(self, phonemes_filename, encoding):
        '''
        Returns a set of the phonemes listed in the given file.
        Assumes the given phonemes_filename is from the CMU Pronouncing Dictionary.
        '''
        phonemes_dictionary = set()
        with codecs.open(phonemes_filename, 'r', encoding) as f:
            # each line is of the form 'UH\tvowel'
            for line in f:
                line = line[:-1] # remove '\n' from end of line
                phoneme_and_type = line.split('\t', 1)
                phonemes_dictionary.add(phoneme_and_type[0])
        return phonemes_dictionary

    def _strip_phonemes_stress(self, phonemes_list):
        '''
        Given a list of list of phonemes, strips out any stress markers (trailing digit 0, 1, or 2).
        Ex., given [['AH0', 'B', 'Y', 'UW', 'Z']], returns [['AH', 'B', 'Y', 'UW', 'Z']].
        '''
        return [[phoneme[:-1] if phoneme[-1].isnumeric() else phoneme for phoneme in phoneme_list] 
                for phoneme_list in phonemes_list]
