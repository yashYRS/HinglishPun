# -*- coding: utf-8 -*-
import csv
import tqdm
import pandas as pd
import epitran
from thefuzz import fuzz


vowels_dict = {k: v for k,v in csv.reader(open('data/svar.csv', 'r'))}
consonants_dict = {k: v for k,v in csv.reader(open('data/vyanjan.csv', 'r'))}


def read_file(file_name):
    with open(file_name) as f:
        data = f.readlines()
    return data


def devng_to_latin_word(devng_word, vowels, consonants):
    consonants_hi, vowels_hi = consonants.keys(), vowels.keys()
    latin_word, len_word = '', len(devng_word)
        
    for i in range(len_word):
        # Check if the vowel markers are present right after the character
        # if vowel present, combine the marker along with the character
        # Note: eg: char, vowel_marker, actual_vowel. Hence in this step, curr_char = char + vowel_marker
        lookahead = 2 if ((i+1) < len_word and devng_word[i + 1].strip()) =='़' else 1
        curr_char = devng_word[i: i+lookahead]

        if curr_char in vowels_hi:
            latin_word += vowels[curr_char]

        elif curr_char in consonants_hi:
            if (i+lookahead) < len_word and devng_word[i+lookahead] in consonants_hi:
                # If the devanagari character after the current one is a consonant
                if (i!=0 and (i+lookahead+1) < len_word and devng_word[i+lookahead+1] in vowels_hi):
                    # We only enter here, if the lookahead was 1, i.e. the curr character didn't have a vowel marker
                    # Check, whether the next consonant is attached to a vowel, if it is, then we don't add 'a' sound
                    latin_word += consonants[curr_char]
                else:
                    latin_word += consonants[curr_char] + 'a'
            else:
                # If the devanagri character after the current one is a vowel
                # Simply add the required latin consonant as is
                # As the special case of 'a' doesn't apply
                latin_word += consonants[curr_char]
    return latin_word


def get_similar_ipas(ipas_hi, ipa_en, thresh):
    return [(word_hi, ipa_hi) for word_hi, ipa_hi in ipas_hi.items() if fuzz.ratio(ipa_en, ipa_hi) > thresh]


def get_homophone_english(word_en, ipas_hi, epi_en, thresh=75):
    ipa_en = epi_en.transliterate(word_en)
    words_info = get_similar_ipas(ipas_hi, ipa_en, thresh)

    words_hi = [i for i,_ in words_info]
    ipas_hi = [j for _,j in words_info]

    return (word_en, ipa_en, words_hi, ipas_hi)


def get_homophone_df(words_hi, words_en, vowels_dict, consonants_dict):

    epi_en = epitran.Epitran('eng-Latn')
    epi_hi = epitran.Epitran('hin-Deva')    
    
    print(" Convert hindi to IPA ")
    ipas_hi = {w: epi_hi.transliterate(w) for w in tqdm.tqdm(words_hi)}
    
    print(" Convert english to IPA and find similar hindi words")
    homophones_list = [get_homophone_english(w, ipas_hi, epi_en) for w in tqdm.tqdm(words_en)]
    # Convert homophone list to a dataframe for convenience
    ipas_df = pd.DataFrame(homophones_list, columns=['en', 'ipa_en', 'hindi_words', 'ipas_hi'])

    # Filter out english words that don't have any similar hindi phonemes
    ipas_df = ipas_df[ipas_df['hindi_words'].apply(lambda x: len(x) > 0)]
    # Per hindi word, transliterate it to the latin
    ipas_df['latin_hindi'] = ipas_df['hindi_words'].apply(
        lambda x: [devng_to_latin_word(i, vowels_dict, consonants_dict) for i in x])    
    return ipas_df


def get_words_from_common():
    df = pd.read_csv('data/hindi_common.csv')
    hindi_words = df['word'].tolist()

    with open('data/english_common.txt') as f:
        english_words = f.readlines()
    english_words = [w.replace('\n', '') for w in english_words]
    return english_words, hindi_words


def get_all_phonemes(vowels_dict, consonants_dict):
    words_en, words_hi = get_words_from_common()
    ipas_df = get_homophone_df(words_hi, words_en, vowels_dict, consonants_dict)

    print(ipas_df.shape)
    ipas_df.to_csv('alternate.csv')


if __name__ == '__main__':
    get_all_phonemes(vowels_dict, consonants_dict)
