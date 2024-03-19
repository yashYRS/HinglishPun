# -*- coding: utf-8 -*-
import csv
import tqdm
from src import utils

import epitran
import pandas as pd

from typing import Union
from pathlib import Path
from deep_translator import GoogleTranslator



script_vowel_file = 'data/transliterate/svar.csv'
script_consonant_file = 'data/transliterate/vyanjan.csv'


class HomophoneGenerator:

    def __init__(self, load_df_file: Union[bool, Path]=False,
                 save_df_file: Union[bool, Path]=False, ipa_sim_thresh: int=75) -> None:

        if isinstance(load_df_file, Path):
            # If homophones have already been generated, load the file, instead of generating everything from scratch
            self.homophone_df: pd.DataFrame = pd.read_csv(load_df_file)

        else:            
            # Get list of common hindi and english words from a corpus
            words_en, words_hi = utils.retrieve_common_words()
            self.words_en: list = words_en
            self.words_hi: list = words_hi

            # Google Translator object
            self.translator = GoogleTranslator(source = 'hi', dest = 'en')

            # Epitran Objects that will be used to convert words to their Ipas
            self.epi_en = epitran.Epitran('eng-Latn')
            self.epi_hi = epitran.Epitran('hin-Deva')

            # Threshold beyond which 2 ipas are considered homophones
            self.ipa_sim_thresh: int = ipa_sim_thresh

            # Dictionaries mapping from Devanagri to script in English
            self.vowels_dict: dict = {k: v for k,v in csv.reader(open(script_vowel_file, 'r'))}
            self.consonants_dict: dict = {k: v for k,v in csv.reader(open(script_consonant_file, 'r'))}

            # Save the possible consonants and vowels as sets to enable faster computation later
            self.vowels_hi: set = set(self.vowels_dict.keys())
            self.consonants_hi: set = set(self.consonants_dict.keys())
            
            self.save_df_file: Path = save_df_file
            self.homophone_df: pd.DataFrame
    
    def get_homophones_df(self) -> None:
        # Retrieve object variables to avoid constant access costs
        thresh, translator = self.ipa_sim_thresh, self.translator
        epi_hi, epi_en = self.epi_hi, self.epi_en
        words_hi, words_en = self.words_hi, self.words_en
        
        # Maps hindi words to their transliterated IPA format
        ipas_hi = {w: epi_hi.transliterate(w) for w in tqdm.tqdm(words_hi)}

        # Per English word, convert it to IPA, and find correspondingly similar Hindi Ipas 
        homophones_list = [utils.get_homophone_en_hi(w, ipas_hi, epi_en, thresh) for w in tqdm.tqdm(words_en)]

        # Convert homophone list to a dataframe for convenience
        homophone_df = pd.DataFrame(homophones_list, columns=['en', 'dvng_hi'])

        # Filter out english words that don't have any similar hindi phonemes
        homophone_df = homophone_df[homophone_df['dvng_hi'].apply(lambda x: len(x) > 0)]

        # Per hindi word, transliterate it to the latin
        homophone_df['latin_hi'] = homophone_df['dvng_hi'].apply(lambda x: [self.devng_to_latin_word(i) for i in x])

        # Get the corresponding translated English words for each of the hindi words
        homophone_df['translated_hi_en'] = homophone_df['dvng_hi'].apply(lambda x: [translator.translate(i) for i in x])

        self.homophone_df = homophone_df


    def save_homophone_df(self) -> None:
        # Save the homopohne datafrae to the csv file
        self.homophone_df.to_csv(self.save_df_file)

    def devng_to_latin_word(self, devng_word: str) -> str:
        # Retrieve object variables to avoid constant access costs
        consonants_hi, vowels_hi = self.consonants_hi, self.vowels_hi
        consonants_dict, vowels_dict = self.consonants_dict, self.vowels_dict
        latin_word, len_word = '', len(devng_word)
            
        for i in range(len_word):
            # Check if the vowel markers are present right after the character
            # if vowel present, combine the marker along with the character
            # Note: eg: char, vowel_marker, actual_vowel. Hence in this step, curr_char = char + vowel_marker
            lookahead = 2 if ((i+1) < len_word and devng_word[i + 1].strip()) =='़' else 1
            curr_char = devng_word[i: i+lookahead]

            if curr_char in vowels_hi:
                latin_word += vowels_dict[curr_char]

            elif curr_char in consonants_hi:
                if (i+lookahead) < len_word and devng_word[i+lookahead] in consonants_hi:
                    # If the devanagari character after the current one is a consonant
                    if (i!=0 and (i+lookahead+1) < len_word and devng_word[i+lookahead+1] in vowels_hi):
                        # We only enter here, if the lookahead was 1, i.e. the curr character didn't have a vowel marker
                        # Check, whether the next consonant is attached to a vowel, if it is, then we don't add 'a' sound
                        latin_word += consonants_dict[curr_char]
                    else:
                        latin_word += consonants_dict[curr_char] + 'a'
                else:
                    # If the devanagri character after the current one is a vowel
                    # Simply add the required latin consonant as is
                    # As the special case of 'a' doesn't apply
                    latin_word += consonants_dict[curr_char]
        return latin_word

    def get_sentences_per_en(self) -> None:
        # Get the list of english words that have homophones
        en_words = self.homophone_df['en'].tolist()

        # Get sentences per en word from the brown corpus where the word appears closer to the end  
        en_candidates = utils.filter_corpus_sentences(en_words)

        # Save the candidates received back to the dataframe
        self.homophone_df['candidate_sentences'] = self.homophone_df['en'].apply(lambda x: en_candidates[x])
