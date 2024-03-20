import tqdm
import nltk
import pandas as pd
from thefuzz import fuzz
from pathlib import Path
from nltk.corpus import brown
from collections import defaultdict


words_file_hi = 'data/words/hindi_common.csv'
words_file_en = 'data/words/english_common.txt'


def is_noun(input_word: str):
    tagged_word = nltk.pos_tag([input_word])
    # Check if the tag of the word starts with 'N' (indicating it's a noun)
    return tagged_word[0][1].startswith('N')


def read_file(file_path: Path):
    with open(file_path) as f:
        data = f.read()
    return data


def write_file(data, file_path: Path):
    with open(file_path, 'w') as f:
        f.write(data)


def retrieve_common_words() -> tuple[list, list]:
    df = pd.read_csv(words_file_hi)
    hindi_words = df['word'].tolist()

    with open(words_file_en) as f:
        english_words = f.readlines()

    brown_words = set(brown.words())
    english_words = [w.replace('\n', '') for w in english_words]
    english_words = [w for w in english_words if len(w) > 2 and w in brown_words]

    return english_words, hindi_words


def get_homophone_en_hi(word_en: list, ipas_hi: dict, epi_en, thresh: int=75):
    ipa_en = epi_en.transliterate(word_en)
    # Apply fuzzy matching to get relatively similar ipas
    words_hi = [(word_hi, fuzz.ratio(ipa_en, ipa_hi)) for word_hi, ipa_hi in ipas_hi.items()]
    # Filter out words that don't match phonetically
    words_hi = [(w, score) for w, score in words_hi if score > thresh]
    # Sort words based on the matching score
    words_hi = sorted(words_hi, key=lambda x: x[1], reverse=True)
    # Remove the scores from the list and just keep the hi word
    words_hi = [w for w, _ in words_hi]
    return (word_en, words_hi)


def filter_corpus_sentences(en_words: list) -> dict:
    all_sentences = brown.sents()
    en_candidates = defaultdict(list)
    
    # Go over all sentences in the corpus
    for curr_sent in tqdm.tqdm(all_sentences):
        # Convert list of words in the sentence to a set of words for faster membership inference
        set_sent = set(curr_sent)
        # Half the length of the sentence stored, to check if word will lie in the 2nd half of the sentence
        half_len = len(curr_sent)/2
        for en_word in en_words:
            # For all the given words, if they appear in the sentence
            # and they appear in the 2nd half of the sentence (towards the end)
            # then that sentence is added to the list of candidate sentences for that word
            # from which puns will be generated
            if en_word in set_sent and curr_sent.index(en_word) > half_len:
                en_candidates[en_word].append(curr_sent)
    return en_candidates

def get_homophone_pairs(df: pd.DataFrame, len_thresh: int=3):
    # Convert homophone pairs to a list of string inputs
    # Filter out as many candidates as possible to minimise LLM costs
    column_list = ['en', 'latin_hi', 'translated_hi_en']
    df = df[column_list]
    
    string_list = []
    for _, row in df.iterrows():
        en_word, hi_list, translated_list = row.en, row.latin_hi, row.translated_hi_en
        added_homophone = False
        for hi_word, hi_translated in zip(hi_list, translated_list):
            if len(hi_word) < len_thresh or added_homophone is True:
                # If the hi word is too short remove or if a homophonic pair has already been added
                # for the given english word then move on to a different english word
                continue
            if fuzz.ratio(hi_translated.lower(), hi_word) > 50:
                # The hi word is most probably a borrowed word, hence skip
                continue
            # The transliterated word is of similar length to the english word & the homophonic word
            # doesn't mean the same thing it does in English
            if abs(len(en_word) - len(hi_word)) < len_thresh and fuzz.ratio(hi_translated.lower(), en_word) < 85:
                added_homophone = True
                # Create the string that will be appended to the prompts
                string_list.append('Input: "{}", "{}" ({})'.format(en_word, hi_word, hi_translated))
    return string_list



def post_process_llm_response(text):
    if 'Output:' in text:
        # Clean the data (chain of thought) and return only the pun in case it was followed
        pun_start = text.index('Output:')
        return text[pun_start:]
    return text


def read_and_clean_tsv(dataset_path):
    df = pd.read_csv(dataset_path, sep = '\t')
    # Push the column to be a row, since the dataset directly starts with
    # transliteration pairs, so push current column names to a row, and add new column names
    df.loc[-1] = df.columns
    df = df.sort_index().reset_index(drop=True)
    df.columns = ['hi','anot_roman']
    return df