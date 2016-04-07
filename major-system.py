# major-system.py
# By Vincent Fiorentini and Megan Shao, (c) 2016.

from pronouncer import Pronouncer

def main():
    '''
    Main function for Major System.
    '''
    pronouncer = Pronouncer()
    print('Phonemes: ' + str(pronouncer.phonemes_set))
    words = ['abuse', 'APPLE', 'Rainbow']
    for word in words:
        print('\'{0}\' (first pronunciation): {1}'.format(word, 
            str(' '.join(pronouncer.pronounce(word)[0]))))
        print('\'{0}\' (all pronunciations): {1}'.format(word, str(pronouncer.pronounce(word))))
        print('\'{0}\' (all pronunciations, with stress): {1}'.format(word, 
            str(pronouncer.pronounce(word, strip_stress = False))))

if __name__ == "__main__":
    main()