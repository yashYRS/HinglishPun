import tqdm
import spacy
import ast
import json
from pathlib import Path

from nltk import pos_tag
from openai import OpenAI

from gensim import downloader as api

from collections import defaultdict

from src import utils
from src.homophone_gen import HomophoneGenerator


class PunGenerator:

    def __init__(self, homophone_df_path: Path, load_homophone: bool=False, method='homophone'):
        
        if method == 'homophone':
            self.top_n_context_thresh = 5
            self.spacy_en = spacy.load("en_core_web_sm")
            self.noun_pos_tags = set(['NN', 'NNS', 'NNP', 'NNPS'])

            # Load pre-trained Word2Vec model
            self.embedding_model = api.load("glove-twitter-25")
            self.embedding_vocab : set = set(self.embedding_model.index_to_key)
            
            if load_homophone is True:
                # If homophones have already been generated, load the file 
                self.homophone_gen = HomophoneGenerator(load_df_file=homophone_df_path)
                homophone_df = self.homophone_gen.homophone_df
                homophone_df['candidate_sentences'] = homophone_df['candidate_sentences'].apply(ast.literal_eval)
                homophone_df['dvng_hi'] = homophone_df['dvng_hi'].apply(ast.literal_eval)
                homophone_df['latin_hi'] = homophone_df['latin_hi'].apply(ast.literal_eval)
                homophone_df['translated_hi_en'] = homophone_df['translated_hi_en'].apply(ast.literal_eval)
            else:
                # Create the homophone df from scratch
                self.homophone_gen = HomophoneGenerator(save_df_file=homophone_df_path)
                self.homophone_gen.get_homophones_df()
                self.get_sentences_per_en()
                self.save_homophone_df()

        elif method == 'prompt':
            self.llm_client = OpenAI()

        self.pun_list = []

    def replace_noun_for_pun(self, input_sentence, pos_sequence, word_context_change):
        # Possible to improve this function a bit
        # Extract noun phrases
        sent_nouns = [i for i, p in pos_sequence if p in self.noun_pos_tags]
        pos_sequence = [p for _, p in pos_sequence]
        if len(sent_nouns) == 0:
            return

        # Use the 1st Noun of the sentence
        reqd_noun = sent_nouns[0]
        similar_words = []

        # Check if input word is in the vocabulary
        if word_context_change in self.embedding_vocab:
            # Get most similar words
            similar_words = self.embedding_model.most_similar(word_context_change, topn=self.top_n_context_thresh)

            # Return the first similar word
            for reqd_word, _ in similar_words:
                if utils.is_noun(reqd_word) is True:
                    # Replace the first noun phrase in the input sentence
                    new_sentence = [i if i != reqd_noun else reqd_word for i in input_sentence]
                    # Add the generated sentence to the pun list
                    self.pun_list.append(' '.join(new_sentence))

    def get_pun_from_translated_context(self):

        for _, row in tqdm.tqdm(self.homophone_gen.homophone_df.iterrows()):
            # For all the sentences where the row 'en' word appears at the end, try to create puns
            # by replace the noun at the start with a potential context word of the associated homophone word
            for input_sentence in row.candidate_sentences:
                if len(input_sentence) > 25:
                    continue

                for hi_word, translate_word in zip(row.dvng_hi, row.translated_hi_en):
                    # If the meanings of the homophones are the same, don't create the pun
                    if translate_word.lower() == row.en:
                        continue
                    new_sentence = [i if i != row.en else hi_word for i in input_sentence]
                    if input_sentence == new_sentence:
                        continue
                    new_pos_sequence = pos_tag(new_sentence)
                    self.replace_noun_for_pun(input_sentence, new_pos_sequence, translate_word)

            with open('data/pun_list.json', 'w') as f:
                json.dump(self.pun_list, f)

    def clean_llm_results():
        pass

    def prompting_llm(self, method='fewshot'):
        if method == 'zeroshot':
            prompt_folder = Path('prompts/zeroshot')
        elif method == 'oneshot':
            prompt_folder = Path('prompts/oneshot')
        else:
            prompt_folder = Path('prompts/fewshot')
        
        prompt_instruct_folder = prompt_folder / 'input'
        prompt_result_folder = prompt_folder / 'results'
        
        for f in prompt_instruct_folder.iterdir():
            # Read the prompt to be given to the LLM
            message_prompt = utils.read_file(f) 
            # Call the LLM Api to get the required response
            # Temperature is high, since we want more deterministic outputs
            response = self.llm_client.chat.completions.create(
                messages=[
                        {
                            "role": "user",
                            "content": message_prompt
                        }
                ],
                model="gpt-3.5-turbo",
                temperature=0.3,
                n=1,
            )
            # Retrieve the prompt result from the LLM
            response = response.choices[0].text.strip()

            # Remove the extra details and just keep the pun
            required_pun = utils.post_process_llm_response(response)
            # Add the generated pun to the list of puns generated by the system
            self.pun_list.append(required_pun)

            result_file = prompt_result_folder / f.name
            
            if result_file.exists():
                # Append the new prompt result to the result file
                existing_data = utils.read_file(result_file)
                response = existing_data + '\n' + response

            # Write the raw response received from the LLM to the result files
            utils.write_file(response, result_file)
