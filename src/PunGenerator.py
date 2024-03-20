import os
import tqdm
import ast
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from nltk import pos_tag
from openai import OpenAI

from gensim import downloader as api

from src import utils
from src.homophone_gen import HomophoneGenerator


class PunGenerator:

    def __init__(self, homophone_df_path: Path, load_homophone: bool=False, method: str='homophone', llm_model_name: str="gpt-3.5-turbo"):
        
        if method == 'homophone':
            self.top_n_context_thresh: int = 5
            self.noun_pos_tags: set = set(['NN', 'NNS', 'NNP', 'NNPS'])

            # Load pre-trained Word2Vec model
            self.embedding_model = api.load("glove-twitter-25")
            self.embedding_vocab : set = set(self.embedding_model.index_to_key)
            self.setup_homophone_df(load_homophone, homophone_df_path, generate_sentence=True)

        elif method == 'prompt':
            self.setup_llm_client(llm_model_name)
        
        elif method == 'homophone_prompt':
            self.setup_homophone_df(load_homophone, homophone_df_path)
            self.setup_llm_client(llm_model_name)

        self.method: str = method 
        self.pun_list: list = []
    
    def setup_homophone_df(self, load_homophone: bool, homophone_df_path: Path, generate_sentence: bool=False):
        if load_homophone is True:
            # If homophones have already been generated, load the file 
            self.homophone_gen = HomophoneGenerator(load_df_file=homophone_df_path)
            homophone_df = self.homophone_gen.homophone_df
            if 'candidate_sentences' in homophone_df.columns:
                homophone_df['candidate_sentences'] = homophone_df['candidate_sentences'].apply(ast.literal_eval)
            homophone_df['dvng_hi'] = homophone_df['dvng_hi'].apply(ast.literal_eval)
            homophone_df['latin_hi'] = homophone_df['latin_hi'].apply(ast.literal_eval)
            homophone_df['translated_hi_en'] = homophone_df['translated_hi_en'].apply(ast.literal_eval)
        else:
            # Create the homophone df from scratch
            self.homophone_gen = HomophoneGenerator(save_df_file=homophone_df_path)
            self.homophone_gen.get_homophones_df()
            if generate_sentence is True:
                self.homophone_gen.get_sentences_per_en()
            self.homophone_gen.save_homophone_df()        

    def setup_llm_client(self, llm_model_name):
        self.llm_model_name: str = llm_model_name
        # Path to the .env file in the base folder
        load_dotenv(Path().absolute() / '.env')
        api_key_value = os.getenv("OPENAPI_KEY")
        self.llm_client = OpenAI(api_key=api_key_value)

    def replace_noun_for_pun(self, input_sentence: list, pos_sequence: list, word_context_change: str):
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

    def clean_llm_results(self):
        pass
    
    def execute_llm_call(self, message_prompt):
        # Call the LLM Api to get the required response
        # Temperature is high, since we want more deterministic outputs
        response = self.llm_client.chat.completions.create(
            messages=[
                    {
                        "role": "user",
                        "content": message_prompt
                    }
            ],
            model=self.llm_model_name,
            temperature=0.3,
            n=1,
        )
        # Retrieve the prompt result from the LLM
        response = response.choices[0].message.content.strip()
        return response

    def prompting_llm_algorithmic(self, prompt_folder: Path=None):
        # Zero shot / One shot / Few shot
        for prompt_type_dir in prompt_folder.iterdir():
            prompt_instruct_folder = prompt_type_dir / 'input'
            prompt_result_folder = prompt_type_dir / 'results'
        
            for input_file in prompt_instruct_folder.iterdir():
                result_file = prompt_result_folder / input_file.name
                # Read the prompt to be given to the LLM
                message_prompt = utils.read_file(input_file)
                # Call the LLM model 
                response = self.execute_llm_call(message_prompt)
                # Remove the extra details and just keep the pun
                required_pun = utils.post_process_llm_response(response)
                # Add the generated pun to the list of puns generated by the system
                self.pun_list.append(required_pun)
                
                if result_file.exists():
                    # Append the new prompt result to the result file
                    existing_data = utils.read_file(result_file)
                    response = existing_data + '\n' + response

                # Write the raw response received from the LLM to the result files
                utils.write_file(response, result_file)

    def prompting_llm_homophone(self, prompt_folder: Path=None, limit_calls: int=10):

        prompt_instruct_folder = prompt_folder / 'input'
        prompt_result_folder = prompt_folder / 'results'

        homophone_df = self.homophone_gen.homophone_df
        # From the dataframe format, get homphones as strings that can be used directly as inputs in our prompts
        homophone_strings = utils.get_homophone_pairs(homophone_df)
        
        # For each of the prompts, query the model with a bunch of homophones
        for input_file in prompt_instruct_folder.iterdir():
            
            result_file = prompt_result_folder / input_file.name
            # Read the prompt to be given to the LLM
            message_prompt = utils.read_file(input_file)
            # Per Prompt type, limit the number of homophone inputs
            homophone_strings = random.sample(homophone_strings, limit_calls)
            
            for homophone_inp in homophone_strings:
                input_prompt = message_prompt + '\n' + homophone_inp
                
                # Call the LLM model
                response = self.execute_llm_call(input_prompt)
                # # Remove the extra details and just keep the pun
                required_pun = utils.post_process_llm_response(response)
                # # Add the generated pun to the list of puns generated by the system
                self.pun_list.append(required_pun)
                
                # Add the homophone given as input during the prompt
                response = homophone_inp + '\n' + response
                if result_file.exists():
                    # Append the new prompt result to the result file
                    existing_data = utils.read_file(result_file)
                    response = existing_data + '\n' + response

                # Write the raw response received from the LLM to the result files
                utils.write_file(response, result_file)
    
    def get_puns(self, prompt_folder: Path=None) -> list:
        if self.method == 'prompt':
            self.prompting_llm_algorithmic(prompt_folder)
            self.clean_llm_results()
        
        elif self.method == 'homophone_prompt':
            self.prompting_llm_homophone(prompt_folder)
            self.clean_llm_results()

        elif self.method == 'homophone':
            self.get_pun_from_translated_context()

        return self.pun_list
