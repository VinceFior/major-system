# number_encoder.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer # note: we could instead use nltk.corpus.cmudict
from ngram_model import NgramModel
from random import sample # for RandomGreedyEncoder
from itertools import product # for RandomGreedyEncoder
from nltk.corpus import brown # for RandomGreedyEncoder
from collections import Counter # for RandomGreedyEncoder
from numpy.random import choice # for NgramContextEncoder
from math import exp # for NgramContextEncoder
from nltk.tag import UnigramTagger # for NgramPOSContextEncoder
from ngram_model import NgramPOSModel # for NgramPOSContextEncoder
from stat_parser import Parser # for ParserEncoder
from ngram_evaluator import NgramEvaluator # for ParserEncoder
from nltk.tag import UnigramTagger # for SentenceTaggerEncoder
from copy import deepcopy # for SentenceTaggerEncoder

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

    def _encode_number_chunk(self, number_chunk):
        '''
        Helper method, such as for encode_number(), that returns a list of all single-word
        encodings for the given number_chunk.
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

    def format_encoding(self, encoding):
        '''
        Given a list of strings (that are an encoding for a number), formats them with proper
        capitalization and spacing.
        '''
        result = ''
        is_sentence_start = True
        for i, word in enumerate(encoding):
            if is_sentence_start:
                result += word.capitalize()
                is_sentence_start = False
            else:
                result += word
            if not any(c.isalpha() for c in word):
                # we assume a punctuation mark indicates the end of the sentence
                is_sentence_start = True
            # add a space only if the next word is not punctuation
            if i + 1 < len(encoding) and any(c.isalpha() for c in encoding[i + 1]):
                result += ' '
        return result

    def decode_words(self, words):
        '''
        Decodes the given list of words as a number (string of digits).
        '''
        return ''.join([self.decode_word(word) for word in words])

    def decode_word(self, word):
        '''
        Decodes the given word to a list of digits (length-one strings).
        '''
        if not any(c.isalpha() for c in word):
            return ''
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

    def _get_phonemes_to_words_dict(self, vocabulary = None):
        '''
        Helper method that returns a dictionary that maps every sequence (tuple) of phonemes (that
        are included in self.phoneme_to_digit_dict) to a list of all words in
        self.pronouncer.pronunciation_dictionary that can be pronounced as that sequence.
        Only includes keys with values (i.e., only includes sequences that actually appear).
        '''
        phonemes_to_words_dict = dict()
        for word, pronunciations in self.pronouncer.pronunciation_dictionary.items():
            pronunciations = self.pronouncer.strip_phonemes_stress(pronunciations)
            # if this word is not in the vocabulary, do not include this word
            if vocabulary != None and word.lower() not in vocabulary:
                continue
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

    def _get_vocab(self, max_vocab_size):
        '''
        Helper method that returns a set of max_vocab_size most common words from the brown corpus.
        '''
        cmu_words = set([word.lower() for word in self.pronouncer.pronunciation_dictionary.keys()])
        words = []
        for category in brown.categories():
            text = brown.words(categories=category)
            words += [word.lower() for word in list(text) if word in cmu_words]
        vocabulary = set([word for word, count in Counter(words).most_common()])
        return vocabulary

    def __repr__(self):
        '''
        We represent (print) any NumberEncoder simply by printing its class name, for readability.
        '''
        return self.__class__.__name__

class GreedyEncoder(NumberEncoder):
    '''
    GreedyEncoder is the base class for any encoder that greedily tries to make as long words
    as possible (within an optional limit) and somehow picks from the set of longest words.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = 2):
        '''
        Initializes the GreedyEncoder. The encoder will greedily group digits to make words as long
        as possible (where the length of a word refers to the number of digits it encodes).
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

    def _select_encoding(self, encodings):
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
        selects a word of that length via self._select_encoding().
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
                    chunk_encoding = self._select_encoding(chunk_encodings)
                    has_found_encoding = True
                    encodings += [chunk_encoding]
                    encoded_index += chunk_length
                else:
                    chunk_length -= 1
            if not has_found_encoding:
                number_chunk = number[encoded_index : encoded_index + max_word_length]
                print('Cannot find encoding for number chunk \'{0}\''.format(number_chunk))
                return None
        return encodings
        # return ' '.join(encodings)

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
    RandomGreedyEncoder is a GreedyEncoder that selects which encoding from the vocabulary to use
    randomly.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = 2, max_vocab_size = None):
        '''
        Initializes the RandomGreedyEncoder. The encoder will greedily group digits to make words
        as long as possible (where the length of a word refers to the number of digits it encodes).
        The randomly selected word will be from the max_vocab_size most common words in both the 
        CMU list and the Brown corpus.
        '''
        super(RandomGreedyEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict, max_word_length = max_word_length)

        if max_vocab_size != None:
            vocabulary = self._get_vocab(max_vocab_size)
            self.phonemes_to_words_dict = self._get_phonemes_to_words_dict(vocabulary)

    def _select_encoding(self, encodings):
        '''
        Randomly selects one encoding from the given encodings.
        '''
        return sample(encodings, 1)[0]


class UnigramGreedyEncoder(GreedyEncoder):
    '''
    UnigramGreedyEncoder is a GreedyEncoder that selects the most common encoding according to a
    unigram model.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = 2):
        '''
        Initializes the UnigramGreedyEncoder. The encoder will greedily group digits to make words
        as long as possible (where the length of a word refers to the number of digits it encodes).
        '''
        super(UnigramGreedyEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict, max_word_length = max_word_length)

        self.unigram = NgramModel(1)

    def _select_encoding(self, encodings):
        '''
        Selects the most common encoding according to unigram probabilities.
        '''
        max_prob = -float('inf')
        max_prob_encoding = None
        for encoding in encodings:
            prob = self.unigram.prob((), encoding)
            if prob > max_prob:
                max_prob = prob
                max_prob_encoding = encoding
        return max_prob_encoding


class ContextEncoder(NumberEncoder):
    '''
    ContextEncoder is the base class for any encoder that uses the previously chosen words as
    context and somehow picks from the set of all possible words (possibly of limited word length).
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = None, context_length = 2, min_sentence_length = 5):
        '''
        Initializes the ContextEncoder. The encoder will consider the context of at most
        context_length previous words to choose the best subsequent word.
        '''
        super(ContextEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict)
        # with the default encoding, the maximum number of digits in any word is 19
        max_digits_per_word = 19
        # if we were given an unusual max_word_length, set it to max_digits_per_word
        if max_word_length == None or max_word_length < 1 or max_word_length > max_digits_per_word:
            max_word_length = max_digits_per_word
        self.max_word_length = max_word_length

        max_reasonable_context = 5 # this limit is arbitrary; 5 was chosen with n-grams in mind
        # if we were given an unusual context_length, set it to max_reasonable_context
        if context_length == None or context_length < 0 or context_length > max_reasonable_context:
            context_length = max_reasonable_context
        self.context_length = context_length

        # to aid encode_number(), we set up a mapping from phoneme sequences to words
        self.phonemes_to_words_dict = self._get_phonemes_to_words_dict()

        # minimum number of words required per sentence
        self.min_sentence_length = min_sentence_length

    def _select_encoding(self, previous_words, encodings):
        '''
        Selects one encoding from the given list of possible encodings, given a tuple of
        previous_words strings as context.
        '''
        # this method is what really distinguishes one context encoder from another
        raise NotImplementedError()

    def encode_number(self, number, max_word_length = None, context_length = None):
        '''
        Encodes the given number (string of digits) as a series of words. Considers all possible
        digit chunks up to max_word_length (if given, otherwise self.max_word_length). Based on the
        previous context_length (if given, otherwise self.context_length) words, the encoder
        selects a word via self._select_encoding().
        '''
        # if not given max_word_length, use class default
        if max_word_length == None:
            max_word_length = self.max_word_length

        # if not given context_length, use class default
        if context_length == None:
            context_length = self.context_length

        encoded_index = 0 # the last index of number we've encoded, inclusive
        encodings = []
        while encoded_index < len(number):
            # for all possible chunks starting at this position, find all possible encodings
            chunk_encodings = set()
            for chunk_length in range(1, max_word_length + 1):
                number_chunk = number[encoded_index : encoded_index + chunk_length]
                chunk_encodings |= set(self._encode_number_chunk(number_chunk))
            # check that there are possible encodings to choose from
            if len(chunk_encodings) == 0:
                number_chunk = number[encoded_index : encoded_index + max_word_length]
                print('Cannot find encoding for number chunk \'{0}\''.format(number_chunk))
                return None
            # add possibility of ending the sentence
            if len(encodings) >= self.min_sentence_length and \
                '.' not in encodings[-self.min_sentence_length:]:
                chunk_encodings |= set('.')
            # select the best encoding from chunk_encodings
            context = tuple(encodings[len(encodings) - context_length : len(encodings)])
            chunk_encoding = self._select_encoding(context, list(chunk_encodings))
            encodings += [chunk_encoding]
            # increment encoded_index based on the chosen chunk_encoding
            encoded_index += len(self.decode_word(chunk_encoding))

        return encodings


class NgramContextEncoder(ContextEncoder):
    '''
    NgramContextEncoder is a ContextEncoder that selects the most common encoding according to an
    n-gram model.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = None, min_sentence_length = 5, n = 3, alpha = 0.1,
        select_most_likely = True):
        '''
        Initializes the NgramContextEncoder. The encoder will consider the context of at most
        (n - 1) previous words and choose the subsequent word with highest n-gram probability
        if select_most_likely is True (otherwise, will sample weighted by probabilities).
        '''
        super(NgramContextEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict, max_word_length = max_word_length,
            context_length = n - 1, min_sentence_length = min_sentence_length)
        self.ngram = NgramModel(n, alpha = alpha)
        self.select_most_likely = select_most_likely

    def _select_encoding(self, previous_words, encodings):
        '''
        Selects the most common encoding according to n-gram probabilities.
        '''
        if len(encodings) == 0:
            return None
        if self.select_most_likely:
            max_prob = -float('inf')
            max_prob_encoding = None
            for encoding in encodings:
                prob = self.ngram.prob(previous_words, encoding)
                if prob > max_prob:
                    max_prob = prob
                    max_prob_encoding = encoding
            return max_prob_encoding
        else:
            probabilities = [exp(self.ngram.prob(previous_words, encoding))
                             for encoding in encodings]
            probability_sum = sum(probabilities)
            probabilities_norm = [probability / probability_sum for probability in probabilities]
            return choice(encodings, p = probabilities_norm)


class NgramPOSContextEncoder(ContextEncoder):
    '''
    NgramPOSContextEncoder is a ContextEncoder that selects the most common encoding according to 
    an n-gram model for parts-of-speech and words.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = None, min_sentence_length = 5, n = 3, alpha = 0.1,
        select_most_likely = True, tagger = UnigramTagger(brown.tagged_sents())):
        '''
        Initializes the NgramPOSContextEncoder. The encoder will consider the context of at most
        (n - 1) previous words and choose the subsequent word with highest probability 
        part-of-speech and with highest n-gram probability if select_most_likely is True 
        (otherwise, will sample weighted by probabilities).
        '''
        super(NgramPOSContextEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict, max_word_length = max_word_length,
            context_length = n - 1, min_sentence_length = min_sentence_length)
        self.ngram = NgramModel(n, alpha = alpha)
        self.select_most_likely = select_most_likely

        self.tagger = tagger
        self.pos_ngram = NgramPOSModel(n, alpha = alpha)

    def _select_encoding(self, previous_words, encodings):
        '''
        Finds the most common part-of-speech and selects the most common encoding with that
        part-of-speech according to n-gram probabilities.
        '''
        previous_pos = tuple([tag for word, tag in self.tagger.tag(previous_words)])
        tag_to_words_dict = {}
        for word, tag in self.tagger.tag(encodings):
            if tag == None:
                continue
            if tag in tag_to_words_dict:
                tag_to_words_dict[tag] += [word]
            else:
                tag_to_words_dict[tag] = [word]

        if self.select_most_likely:
            return self._select_most_likely_encoding(previous_words, previous_pos,
                tag_to_words_dict, encodings)
        else:
            return self._select_weighted_prob_encoding(previous_words, previous_pos,
                tag_to_words_dict, encodings)

    def _select_most_likely_encoding(self, previous_words, previous_pos, tag_to_words_dict,
        encodings):
        # find most likely part-of-speech for the next word
        max_prob = -float('inf')
        max_prob_tag = None
        for tag in tag_to_words_dict.keys():
            prob = self.pos_ngram.prob(previous_pos, tag)
            if prob > max_prob:
                max_prob = prob
                max_prob_tag = tag
        # find most likely possible next word that has most likely POS tag
        if max_prob_tag != None:
            encodings = tag_to_words_dict[max_prob_tag]
        max_prob = -float('inf')
        max_prob_encoding = None
        for encoding in encodings:
            prob = self.ngram.prob(previous_words, encoding)
            if prob > max_prob:
                max_prob = prob
                max_prob_encoding = encoding
        return max_prob_encoding

    def _select_weighted_prob_encoding(self, previous_words, previous_pos, tag_to_words_dict,
        encodings):
        # select POS tag for next word from weighted probabilities
        tags = list(tag_to_words_dict.keys())
        pos_probabilities = [exp(self.pos_ngram.prob(previous_pos, tag)) for tag in tags]
        pos_probability_sum = sum(pos_probabilities)
        pos_probabilities_norm = [prob / pos_probability_sum for prob in pos_probabilities]
        tag = choice(tags, p = pos_probabilities_norm)
        # select next word with POS chosen above from weighted probabilties
        encodings = tag_to_words_dict[tag]
        probabilities = [exp(self.ngram.prob(previous_words, encoding)) for encoding in encodings]
        probability_sum = sum(probabilities)
        probabilities_norm = [probability / probability_sum for probability in probabilities]
        return choice(encodings, p = probabilities_norm)

class ParserEncoder(NumberEncoder):
    '''
    ParserEncoder is a NumberEncoder that creates grammatically plausible sentences.
    '''

    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_vocab_size = 10000, parser = Parser(), evaluator = NgramEvaluator(2)):
        '''
        Initializes the ParserEncoder.
        '''
        super(ParserEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict)
        # set up our size-limited vocab
        if max_vocab_size != None:
            vocabulary = self._get_vocab(max_vocab_size)
            self.phonemes_to_words_dict = self._get_phonemes_to_words_dict(vocabulary)
        else:
            self.phonemes_to_words_dict = self._get_phonemes_to_words_dict()

        self.parser = parser
        self.evaluator = evaluator

    def _encode_number_pos(self, number, pos_tags, pre_context, post_context):
        '''
        Helper method that encodes the given number (string of digits) as the most likely series
        of one or two words that, as a phrase, has a part-of-speech tag in pos_tags. Picks the
        encoding with the highest evaluator score (incorporating the given context, lists of the
        immediately preceding and proceeding words).
        '''
        encodings = []
        max_word_length = len(number)
        # split at each index so the first word has length 1, 2, ..., len(number)
        for split in range(1, max_word_length + 1):
            first_chunk = number[:split]
            second_chunk = number[split:]
            first_encodings = self._encode_number_chunk(first_chunk)
            second_encodings = self._encode_number_chunk(second_chunk)
            if split == max_word_length:
                encodings += [[first_encodings]]
            else:
                encodings += [[first_encodings] + [second_encodings]]

        best_pos_score = -float("inf")
        best_valid_phrase = None
        # as a backup, we also keep track of the highest-scoring encoding of any tag
        best_any_score = -float("inf")
        best_any_phrase = None
        for encoding_split in encodings:
            for enc in product(*encoding_split):
                if len(enc) != 0:
                    enc_with_context = pre_context + list(enc) + post_context
                    score = self.evaluator.score(enc_with_context)
                    if score > best_any_score:
                        best_any_score = score
                        best_any_phrase = enc
                    if score > best_pos_score:
                        tree = self.parser.parse(' '.join(enc))
                        if tree:
                            label = tree.label()
                            if label in pos_tags:
                                best_valid_phrase = enc
                                best_pos_score = score

        # if we found no phrases of the right part of speech, return the highest-scoring phrase
        if best_valid_phrase is None:
            return best_any_phrase
        # TODO: if found no encoding of 1 or 2 words, fall back to another model
        return best_valid_phrase

    def encode_number(self, number):
        '''
        Encodes the given number (string of digits) as a series of words. Does so by making a
        series of sentences of the form "Noun Phrase, Verb Phrase, Noun Phrase," each of which
        encodes 3 digits in one or two words.
        '''
        # todo: review full tag list (in parse.py) and decide how to handle "+"
        noun_tags = ['NP', 'NX', 'NX+NX', 'NX+NP', 'NP+NP', 'RRC',
                     'NN', 'NNS', 'NNP', 'NNPS'] # noun-ish tags (exclude 'SQ')
        verb_tags = ['VP', 'VBZ', 'SQ+VP', 'VP+VP',
                     'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
                     # verb-ish tags (exclude 'VP+PP'?)
        chunk_size = 3
        encodings = []
        for start_index in range(0, len(number), chunk_size):
            chunk = number[start_index : start_index + chunk_size]
            # our sentences are of the form: noun, verb, noun.
            pos_index = (start_index / chunk_size) % 3
            if pos_index == 0:
                pos_tags = noun_tags
            elif pos_index == 1:
                pos_tags = verb_tags
            elif pos_index == 2:
                pos_tags = noun_tags
            pre_context = ['.'] + encodings
            # the last fragment ends with a period as well
            is_sentence_end = (pos_index == 2) or (start_index >= len(number) - chunk_size)
            post_context = ['.'] if is_sentence_end else []
            chunk_encoding = self._encode_number_pos(chunk, pos_tags, pre_context, post_context)
            for word in chunk_encoding:
                encodings += [word]
            if is_sentence_end:
                encodings += ['.']

        return encodings

class SentenceTaggerEncoder(NgramContextEncoder):
    '''
    SentenceTaggerEncoder is an NgramContextEncoder that creates sentences whose part-of-speech
    tags match a sentence in the training corpus, selecting from valid words by n-gram probability.
    '''
    def __init__(self, pronouncer = Pronouncer(), phoneme_to_digit_dict = None,
        max_word_length = None, min_sentence_length = 5, n = 3, alpha = 0.1,
        select_most_likely = True, tagger_type = UnigramTagger, 
        tagged_sents = brown.tagged_sents(tagset='universal'), num_sentence_templates = 100,
        word_length_weight = 10):
        '''
        Initializes the SentenceTaggerEncoder.
        Uses the num_sentence_templates most common part-of-speech sentence types, requiring at
        least min_sentence_length words per sentence. Favors words that encode more digits as a
        function of word_length_weight (0 means unweighted). Scores words and sentences through an
        n-gram model with the given n and alpha.
        '''
        super(SentenceTaggerEncoder, self).__init__(pronouncer = pronouncer,
            phoneme_to_digit_dict = phoneme_to_digit_dict, max_word_length = max_word_length,
            min_sentence_length = min_sentence_length, n = n, alpha = alpha,
            select_most_likely = select_most_likely)

        # set up our tagger and sentence_templates
        self.tagger = tagger_type(tagged_sents)
        self.tagged_sents = tagged_sents
        self.num_sentence_templates = num_sentence_templates
        # some parts of speech can be reasonably omitted from any sentence - we call these optional
        self.optional_tags = ['DET', 'ADJ', 'ADV']
        self.sentence_templates = self._get_sentence_templates()
        self.word_length_weight = word_length_weight

    def _get_sentence_templates(self):
        '''
        Returns a list of tuples containing a sentence template (POS tag tuple) and count.
        For example, an element of the returned list is of the form (('ADJ', 'NOUN'), 85).
        '''
        # extract the POS tags (not the actual words) from self.tagged_sents, skipping punctuation 
        # tags
        sent_tags = [tuple([word_tag[1] for word_tag in tag_sent if word_tag[1] != '.'])
                     for tag_sent in self.tagged_sents]
        # filter out any sentence with an unknown word or a number
        def bad_tag(tag):
           return (tag == 'X' or tag == 'NUM')
        sent_tags_filtered = list(filter(lambda tag_sent:not any(bad_tag(tag) for tag in tag_sent),
                                  sent_tags))
        # filter out any sentence that has no 'VERB'
        sent_tags_filtered = [tag_sent for tag_sent in sent_tags_filtered if 'VERB' in tag_sent]
        # filter out any sentence that does not have enough required parts of speech (is too short)
        def long_enough(tag_sent):
            num_optional_tags = sum([tag_sent.count(tag) for tag in self.optional_tags])
            num_required_tags = len(tag_sent) - num_optional_tags
            return num_required_tags >= self.min_sentence_length
        sent_tags_filtered = [tag_sent for tag_sent in sent_tags_filtered if long_enough(tag_sent)]
        # select only the self.num_sentence_templates most common templates
        templates = Counter(sent_tags_filtered).most_common(self.num_sentence_templates)
        return templates

    def _get_sentence_template(self, templates):
        '''
        Randomly selects a sentence template (a tuple of POS tags) from the given templates.
        Returns a tuple containing the template and its index.
        '''
        probs = []
        total_prob = 0
        for template in templates:
            template_prob = template[1]
            probs += [template_prob]
            total_prob += template_prob
        probs_norm = [prob / total_prob for prob in probs]
        template_sentences = [t[0] for t in templates]
        # choice only works with multiple items, so we manually check if there is only one template
        if len(template_sentences) == 1:
            return (template_sentences[0], 0)
        sentence_template_index = choice(len(template_sentences), p = probs_norm)
        return (template_sentences[sentence_template_index], sentence_template_index)

    def _select_encoding(self, previous_words, encodings):
        '''
        Selects the most common encoding according to n-gram probabilities, weighted toward the
        encoding that encodes the most digits by self.word_length_weight (0 is unweighted).
        '''
        if len(encodings) == 0:
            return None
        elif len(encodings) == 1:
            return encodings[0]
        word_length_weight = self.word_length_weight
        if self.select_most_likely:
            max_prob = -float('inf')
            max_prob_encoding = None
            for encoding in encodings:
                prob = exp(self.ngram.prob(previous_words, encoding))
                weighted_prob = prob * pow(len(self.decode_word(encoding)), word_length_weight)
                if weighted_prob > max_prob:
                    max_prob = weighted_prob
                    max_prob_encoding = encoding
            return max_prob_encoding
        else:
            probabilities = [exp(self.ngram.prob(previous_words, encoding)) *
                             pow(len(self.decode_word(encoding)), word_length_weight)
                             for encoding in encodings]
            probability_sum = sum(probabilities)
            probabilities_norm = [probability / probability_sum for probability in probabilities]
            return choice(encodings, p = probabilities_norm)

    def encode_number(self, number, max_word_length = None, context_length = None, num_times = 1,
        evaluator = None):
        '''
        Generates num_times encodings and returns the encoding with the lowest perplexity according
        to the evaluator (assumes the evaluator has a perplexity() function).
        '''
        if evaluator == None:
            evaluator = NgramEvaluator(self.ngram.n, ngram_model = self.ngram)
        # generate num_times encodings
        encodings = []
        for i in range(num_times):
            encoding = self.encode_number_once(number, max_word_length, context_length)
            encodings += [encoding]
        # if there is only one encoding, don't bother evaluating it
        if len(encodings) == 1:
            return encodings[0]
        # determine encoding with the lowest perplexity - note that all perplexities might be
        # infinite if the encoding is long
        min_perplexity = float('inf')
        min_perplexity_encoding = None
        for encoding in encodings:
            perplexity = evaluator.perplexity(encoding)
            if perplexity <= min_perplexity:
                min_perplexity = perplexity
                min_perplexity_encoding = encoding
        return min_perplexity_encoding

    def encode_number_once(self, number, max_word_length = None, context_length = None):
        '''
        Encodes the given number (string of digits) as a series of words. This series of words
        will be a series of sentences (separated by periods). Considers all possible digit chunks
        up to max_word_length and, based on the previous context_length words, selects a word that
        matches the needed part-of-speech tag via self._select_encoding().
        '''
        # if not given max_word_length, use class default
        if max_word_length == None:
            max_word_length = self.max_word_length

        # if not given context_length, use class default
        if context_length == None:
            context_length = self.context_length

        encoded_index = 0 # the last index of number we've encoded, inclusive
        encodings = []
        sentence_template = None
        templates = deepcopy(self.sentence_templates)
        while encoded_index < len(number):
            if (sentence_template == None) or (sentence_index == len(sentence_template) - 1):
                # if we successfully matched a sentence, we can sample from all templates
                if sentence_template != None:
                    templates = deepcopy(self.sentence_templates)
                sentence_template, sentence_template_index = self._get_sentence_template(templates)
                del templates[sentence_template_index] # since we've used this template, remove it
                sentence_index = 0
                if len(encodings) != 0:
                    encodings += ['.']
            else:
                sentence_index += 1
            # for all possible chunks starting at this position, find all possible encodings
            chunk_encodings = set()
            for chunk_length in range(1, max_word_length + 1):
                number_chunk = number[encoded_index : encoded_index + chunk_length]
                chunk_encodings |= set(self._encode_number_chunk(number_chunk))
            # filter out the chunk_encodings that do not match the needed part-of-speech tag
            pos_tag = sentence_template[sentence_index]
            pos_tags = [pos_tag]
            # if the pos_tag is a pronoun, we allow a noun instead
            if pos_tag == 'PRON':
                pos_tags += ['NOUN']
            chunk_encodings = [encoding for encoding in chunk_encodings
                               if self.tagger.tag([encoding])[0][1] in pos_tags]
            # select the best encoding from chunk_encodings
            context = tuple(encodings[len(encodings) - context_length : len(encodings)])
            # note: we could improve the context by adding a post_context (i.e., a period at end)
            chunk_encoding = self._select_encoding(context, list(chunk_encodings))
            
            if chunk_encoding != None:
                encodings += [chunk_encoding]
                # increment encoded_index based on the chosen chunk_encoding
                encoded_index += len(self.decode_word(chunk_encoding))
            elif pos_tag not in self.optional_tags:
                # if none of the chunk_encodings matches the needed pos_tag, remove all encodings
                # used in the current sentence and select a (hopefully) new sentence template
                sentence_template = None
                if '.' not in encodings:
                    encodings = []
                    encoded_index = 0
                else:
                    last_period_index = (len(encodings) - 1) - encodings[::-1].index('.')
                    partial_sentence = encodings[last_period_index + 1:]
                    encodings = encodings[:last_period_index]
                    partial_sentence_len = len(self.decode_words(partial_sentence))
                    encoded_index -= partial_sentence_len
        encodings += ['.']
        return encodings
