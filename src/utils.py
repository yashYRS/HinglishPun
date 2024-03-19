import tqdm
import nltk
import pandas as pd
from thefuzz import fuzz
from pathlib import Path
from nltk.corpus import brown
from collections import defaultdict


# nltk.download('averaged_perceptron_tagger')
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
    words_hi = [word_hi for word_hi, ipa_hi in ipas_hi.items() if fuzz.ratio(ipa_en, ipa_hi) > thresh]
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


def post_process_llm_response(text):
    # Clean the data (chain of thought) and return only the pun
    pass