import tqdm
import nltk
import json
import epitran
import numpy as np
import pandas as pd

from thefuzz import fuzz
from pathlib import Path

from nltk.corpus import brown
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict




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


def retrieve_common_words(words_file_en: Path, words_file_hi: Path) -> tuple[list, list]:
    """From the common Hindi & English corpuses read the common words
    clean formatting, wherever necessary and return the corresponding lists

    Args:
        words_file_en (pathlib.Path): Path to the corpus containing English words in different lines (csv file)
        words_file_hi (pathlib.Path): Path to the corpus containing hindi words in different lines (txt file)

    Returns:
        tuple[list, list]: List of Common English words, List of Common Hindi words
    """    
    df = pd.read_csv(words_file_hi)
    hindi_words = df['word'].tolist()

    with open(words_file_en) as f:
        english_words = f.readlines()

    # Filter out English words that don't appear in the Brown corpus to be able to 
    # create candidate sentences easily
    brown_words = set(brown.words())
    english_words = [w.replace('\n', '') for w in english_words]
    english_words = [w for w in english_words if len(w) > 2 and w in brown_words]

    return english_words, hindi_words


def get_homophone_en_hi(word_en: str, ipas_hi: dict, epi_en: epitran._epitran.Epitran, thresh: int=75):
    """
    Perform Minimum Edit distance between hindi ipas and english ipa
    Sort the hindi words with decreasing similarity with the English word
    Filter out words where the similarity is below the given threshold

    Args:
        word_en (str): English word for which corresponding hindi homophones need to be found
        ipas_hi (dict): Hindi word to IPA mapping
        epi_en (epitran._epitran.Epitran): Epitran object for English for getting it's IPA
        thresh (int, optional): Minimum Similarity between IPAs to be considered homophonic. Defaults to 75

    Returns:
        str, list: English word, List of hindi words which are homophones of the given English word
    """    
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
    """Select sentences from the Brown corpus where the given english words appear towards the end

    Args:
        en_words (list): List of english words for which sentences need to be found

    Returns:
        dict: word to candidate sentences mapping
    """    
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
    """Query and filter out homophonic pairs from the dataframe and convert it to prompts

    Args:
        df (pd.DataFrame): Dataframe containing homophones
        len_thresh (int, optional): Difference in length between Hindi and English. Defaults to 3.

    Returns:
        list: List of strings that represents the input prompt containing Hindi English homophones 
    """
    # Initialising list of homophone pairs
    homophone_pairs = []
    
    for _, row in df.iterrows():
        # To limit the number of homophones while maintaining diversity
        # if a homophone has been found for a English word, we move on to the next one
        added_homophone = False
        en_word, hi_list, translated_list = row.en, row.latin_hi, row.translated_hi_en
        
        for hi_word, hi_translated in zip(hi_list, translated_list):
        
            if len(hi_word) < len_thresh or added_homophone is True:
                # If the hi word is too short remove or if a homophonic pair has already been added
                # for the given english word then move on to a different english word
                continue
        
            if fuzz.ratio(hi_translated.lower(), hi_word) > 50:
                # The hi word is most probably a borrowed word, hence skip
                continue
        
            # The transliterated word should be of similar length to the english word
            # The homophonic word shouldn't mean the same thing in Hindi as it does in English        
            if abs(len(en_word) - len(hi_word)) < len_thresh and fuzz.ratio(hi_translated.lower(), en_word) < 85:
                # Create the string that will be appended to the prompts
                homophone_pairs.append('Input: "{}", "{}" ({})'.format(en_word, hi_word, hi_translated))
                added_homophone = True

    return homophone_pairs



def post_process_llm_response(text):
    """
    Args:
        text (str): Longer text, containing chain of thought as well as the final pun

    Returns:
        str: Filtered out text, containing only the joke, and nothing else
    """
    if 'Output:' in text:
        pun_start = text.index('Output:')
        return text[pun_start:]
    return ''


def read_and_clean_tsv(dataset_path):
    """Clean the TSV file containing ground truth annotations of transliterations of the 
    Google Research Team - Dakhsina Dataset for evaluating transliteration module

    Args:
        dataset_path (Path): path to the dataset (.tsv) file

    Returns:
        pd.Dataframe: Columns containing the Hindi word and the corresponding ground truth 
            latin transliteration
    """    
    df = pd.read_csv(dataset_path, sep = '\t')

    # Push the column to be a row, since the dataset directly starts with
    # transliteration pairs, so push current column names to a row
    df.loc[-1] = df.columns
    # Reset index due to the presence of the new row
    df = df.sort_index().reset_index(drop=True)
    # Rename columns
    df.columns = ['hi','anot_roman']

    # Filter out rows which contain numbers, symbols and formatting etc.
    df = df[df.anot_roman.str.isalpha() == True]

    return df

def bar_chart(title: str, max_scores_dict: dict, min_scores_dict, mean_scores_dict, file_path: Path):
    """
    Args:
        title (str): title of the box plot
        scores_dict (dict): Maps labels to a list of scores
        file_path (Path): Path where the plot image will be saved
    """
    sns.set_theme(style="whitegrid", palette="bright6", context="paper", font_scale=2)
    # Get the labels, and the values from the dictionary
    categories, max_values = list(max_scores_dict.keys()), list(max_scores_dict.values()), 
    min_values, mean_values = list(min_scores_dict.values()), list(mean_scores_dict.values())
    max_values = [round(v, 2) for v in max_values]
    min_values = [round(v, 2) for v in min_values]
    mean_values = [round(v, 2) for v in mean_values]


    # the label locations
    width = 0.25
    x = np.arange(len(categories))
    fig, ax = plt.subplots()
    _ = ax.bar(x - width, mean_values, width, label='Overall')
    _ = ax.bar(x, max_values, width, label='Top 2')
    _ = ax.bar(x + width, min_values, width, label='Bottom 2')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_xlabel('Categories')
    ax.set_ylabel('Mean Ratings')
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.tick_params(axis='x', labelsize=15)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncols=3, prop={'size': 13})
    fig.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def box_plot(title: str, scores_dict: dict, file_path: Path, remove_human: bool=False):
    """
    Args:
        title (str): title of the box plot
        scores_dict (dict): Maps labels to a list of scores
        file_path (Path): Path where the plot image will be saved
        remove_human (bool, optional): if True, human category removed before plotting. Defaults to False.
    """
    sns.set_theme(style="whitegrid", palette="bright6", context="paper", font_scale=2)
    # Remove human scores, if normalised data is being plotted
    if remove_human is True:
        scores_dict.pop('Human', None)
    # Get the labels, and the values from the dictionary
    categories, values = list(scores_dict.keys()), list(scores_dict.values())
    
    # Create boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(values, labels=categories)

    # Add title and labels
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Ratings')

    # plt.grid(True)
    # plt.savefig(file_path)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.show()


def plot_survey_results(survey_folder: Path, min_human_score: float=3):
    """Visualise some of the results collected during the survey after removing
    users who gave low scores across all human generated jokes 

    Args:
        survey_folder (Path): Path where data to be plotted is stored
        min_human_score (float, optional): Minimum mean score expected by participants. Defaults to 3.
    """    
    survey_map_file = survey_folder / 'survey_map.json'
    survey_file = survey_folder / 'survey_results.csv'
    
    with open(survey_map_file) as f:
        pun_key = json.load(f)

    # Drop Timestamp column as its not needed
    pun_ratings_df = pd.read_csv(survey_file)
    pun_ratings_df = pun_ratings_df.drop('Timestamp', axis=1)
    pun_ratings_df.columns = [c.replace('\"', '') for c in pun_ratings_df.columns]
    
    human_ratings = pun_ratings_df[pun_key['Human']]
    # Get a mapping from the participant index to the average ratings they gave to the human generated jokes
    humans_mean = human_ratings.T.mean(skipna=True).to_dict()

    # Remove the indices of participants which gave extremely low scores to human scores
    valid_rating_dict = {k: value for k, value in humans_mean.items() if value > min_human_score}
    print("Number of valid participants - {}".format(len(valid_rating_dict)))

    category_scores = defaultdict(list)
    category_norm_scores = defaultdict(list)

    for idx, row in pun_ratings_df.iterrows():
        # filter out participants who gave extremely low scores to human generated puns
        if idx not in valid_rating_dict:
            continue

        # Normalising factor (based on ratings given to human puns) for the participant
        normalise_factor = valid_rating_dict[idx]

        for category, pun_list in pun_key.items():
            # Get the mean ratings given to that category by the participant
            mean_score = row[pun_list].mean(skipna=True)
            # Store the scores received per category
            category_scores[category].append(mean_score)
            # Normalise the participant score, based on human ratings given
            normalised_score = (mean_score * normalise_factor)/5
            # Cap rating to be max 5
            category_norm_scores[category].append(min(normalised_score, 5))
    
    temp_file_path = survey_folder / 'raw_ratings_box.pdf'
    box_plot(title='Raw Pun Ratings', scores_dict=category_scores,
             file_path=temp_file_path, remove_human=False)

    temp_file_path = survey_folder / 'norm_ratings_box.pdf'
    box_plot(title='Normalised Pun Ratings Per Category', scores_dict=category_norm_scores,
             file_path=temp_file_path, remove_human=True)

    # Mean scores per category and plot bar chart - works
    # temp_file_path = survey_folder / 'mean_ratings.pdf'
    # mean_scores = {category: sum(scores)/ len(scores) for category, scores in category_scores.items()}

    # # Get Max and Min scores per category and plot bar chats on means of them
    # sorted_scores = {category: sorted(scores, reverse=True) for category, scores in category_scores.items()}
    # min_mean_scores = {category: sum(scores[-2:])/2 for category, scores in sorted_scores.items()}
    # max_mean_scores = {category: sum(scores[:2])/2 for category, scores in sorted_scores.items()}

    # bar_chart('Mean Ratings per Category', max_mean_scores, min_mean_scores, mean_scores, temp_file_path)
    # temp_file_path = survey_folder / 'max_2_mean_ratings.png'
    # bar_chart(title='Mean of the Top 2 Scores per Category', scores_dict=max_mean_scores,
    #               file_path=temp_file_path)
    # temp_file_path = survey_folder / 'min_2_mean_ratings.png'    
    # bar_chart(title='Mean of the Top 2 Scores per Category', scores_dict=min_mean_scores,
     #              file_path=temp_file_path)