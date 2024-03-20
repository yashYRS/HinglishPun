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

    def __init__(self, homophone_df_path: Path, load_homophone: bool=False, method: str='homophone',
                 llm_model_name: str="gpt-3.5-turbo", llm_temperature: float=0.3):
        """
        Args:
            homophone_df_path (Path): Path to the file where the homophone df is either stored or will be stored
            load_homophone (bool, optional): If False, homophones are generated from scratch. Defaults to False.
            method (str, optional): Possible types are `prompting`, `homophone`, `homophone_prompt`. 
                which refer to standard instruction following prompting where the model is supposed to find the
                homophone itself as well, where we generate the homophones and even the subsequent joke without the model
                through GLOve embeddings and finally where we provide the LLM with homophones and ask it to generate
                puns by exploiting them. Defaults to 'homophone'.
            llm_model_name (str, optional): The Open AI model of choice. Defaults to "gpt-3.5-turbo".
            llm_temperature (float, optional): Lower temperature results in more predictable responses. Defaults to 0.3.
        """
        if method == 'homophone':
            # The number of possible replacements that will be used to replace the existing noun phrase
            self.top_n_context_thresh: int = 5
            self.noun_pos_tags: set = set(['NN', 'NNS', 'NNP', 'NNPS'])

            # Load pre-trained Word2Vec model
            self.embedding_model = api.load("glove-twitter-25")
            # Store the entire vocabulary found in the embedding model to allow faster membership inference
            self.embedding_vocab : set = set(self.embedding_model.index_to_key)
            self.setup_homophone_df(load_homophone, homophone_df_path, generate_sentence=True)

        elif method == 'prompt':
            self.setup_llm_client(llm_model_name, llm_temperature)
        
        elif method == 'homophone_prompt':
            self.setup_homophone_df(load_homophone, homophone_df_path)
            self.setup_llm_client(llm_model_name, llm_temperature)

        self.method: str = method 
        self.pun_list: list = []
    
    def setup_homophone_df(self, load_homophone: bool, homophone_df_path: Path, generate_sentence: bool=False):
        """Either load the homophone dataframe or create it from scratch by comparing IPAs of English and Hindi words

        Args:
            load_homophone (bool): If False, homophones are generated from scratch
            homophone_df_path (Path): Path to the file where the homophone df is either stored or will be stored
            generate_sentence (bool, optional):If True, then candidate sentences per English word in our dataframe 
                will be also be stored in our dataframe to aid in constructing puns. Defaults to False.
        """        
        if load_homophone is True:
            # If homophones have already been generated, load the file 
            self.homophone_gen = HomophoneGenerator(load_df_file=homophone_df_path)
            homophone_df = self.homophone_gen.homophone_df

            # Since we are loading the dataframe from a csv, interpret string representation of lists as lists
            if 'candidate_sentences' in homophone_df.columns:
                homophone_df['candidate_sentences'] = homophone_df['candidate_sentences'].apply(ast.literal_eval)

            homophone_df['dvng_hi'] = homophone_df['dvng_hi'].apply(ast.literal_eval)
            homophone_df['latin_hi'] = homophone_df['latin_hi'].apply(ast.literal_eval)
            homophone_df['translated_hi_en'] = homophone_df['translated_hi_en'].apply(ast.literal_eval)
        else:
            # Create the homophone df from scratch
            self.homophone_gen = HomophoneGenerator(save_df_file=homophone_df_path)
            self.homophone_gen.get_homophones_df()
            # Gather candidate sentences per english word
            if generate_sentence is True:
                self.homophone_gen.get_sentences_per_en()
            # Save the homophone df to reuse in subsequent runs
            self.homophone_gen.save_homophone_df()        

    def setup_llm_client(self, llm_model_name: str, llm_temperature: float):
        """Initialise the Open AI model by reading the api key

        Args:
            llm_model_name (str): The Open AI model of choice
            llm_temperature (float): Lower temperature results in more predictable responses
        """        
        self.llm_model_name: str = llm_model_name
        self.llm_temperature: float = llm_temperature

        # Path to the .env file in the base folder
        load_dotenv(Path().absolute() / '.env')
        api_key_value = os.getenv("OPENAPI_KEY")

        # Initialise the OPEN AI model
        self.llm_client = OpenAI(api_key=api_key_value)

    def replace_noun_for_pun(self, input_sentence: list, pos_sequence: list, word_context_change: str):
        """Replace the first noun phrase of the sentence with a corresponding noun phrase that is more
        related to the context of the given input word 

        Args:
            input_sentence (list): Input sentence which needs to be changed
            pos_sequence (list): Pos tagged data of the given sentence 
            word_context_change (str): Word based on which, the starting noun phrase will be changed
        """
        # Extract noun phrases, if no noun found, return as is
        sent_nouns = [i for i, p in pos_sequence if p in self.noun_pos_tags]
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
                # If the similar word also belongs to the same POS tag category as the one its replacing
                if utils.is_noun(reqd_word) is True:
                    # Replace the first noun phrase in the input sentence
                    new_sentence = [i if i != reqd_noun else reqd_word for i in input_sentence]
                    # Add the generated sentence to the pun list
                    self.pun_list.append(' '.join(new_sentence))

    def get_pun_from_translated_context(self):
        """For all the homophones in the dataframe, try and generate puns by manipulating some 
        sentences corresponding to the English word. Many sentences where the english word from the 
        homophone pair lies at the end of the sentence. In such cases, the English word is replaced by its
        homphonic hindi pair, and the starting Noun phrase of the sentence is changed to make the overall
        sentence still make sense. 
        """
        for _, row in tqdm.tqdm(self.homophone_gen.homophone_df.iterrows()):
            # For all the sentences where the row 'en' word appears at the end, try to create puns
            # by replace the noun at the start with a potential context word of the associated homophone word
            for input_sentence in row.candidate_sentences:
                # If the sentence is too long, skip
                if len(input_sentence) > 25:
                    continue

                for hi_word, translate_word in zip(row.dvng_hi, row.translated_hi_en):
                    # If the meanings of the homophones are the same, don't create the pun
                    if translate_word.lower() == row.en:
                        continue
                    # Replace the English word of the homophonic pair with the Hindi word 
                    new_sentence = [i if i != row.en else hi_word for i in input_sentence]
                    # If both words were the same, skip
                    if input_sentence == new_sentence:
                        continue
                    # Tag the sentence to identify the noun phrase at the start
                    pos_sequence = pos_tag(input_sentence)
                    # Replace the Noun Phrase at the start with a more contextual noun
                    self.replace_noun_for_pun(input_sentence, pos_sequence, translate_word)
    
    def execute_llm_call(self, message_prompt):
        """Call the LLM OPENAI client with the given message prompt

        Args:
            message_prompt (str): the Prompt to be passed to the LLM

        Returns:
            str: Response from the LLM
        """        
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
            temperature=self.llm_temperature,
            n=1,
        )
        # Retrieve the prompt result from the LLM
        response = response.choices[0].message.content.strip()
        return response

    def prompting_llm_algorithmic(self, prompt_folder: Path=None):
        """Prompt LLMs by describing a basic algorithm for generating puns

        Args:
            prompt_folder (Path, optional): Path where the prompts are stored. Defaults to None.
        """        
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
        """Prompt LLMs by providing them Homophone pairs as input along with some examples of 
        the kind of puns / one liners we are looking for, given such a homophonic pair

        Args:
            prompt_folder (Path, optional): Path where the prompts are stored. Defaults to None.
            limit_calls (int, optional): Number of homophones to pass per Prompt type. Defaults to 10.
        """
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
        """Wrapper function to provide a unified API for generating puns from this package

        Args:
            prompt_folder (Path, optional): Path to where the prompts are stored. Defaults to None.
                Doesn't need to be passed in case method doesn't involve prompting

        Returns:
            list: List of puns generated
        """        
        if self.method == 'prompt':
            self.prompting_llm_algorithmic(prompt_folder)
            self.clean_llm_results()
        
        elif self.method == 'homophone_prompt':
            self.prompting_llm_homophone(prompt_folder)
            self.clean_llm_results()

        elif self.method == 'homophone':
            self.get_pun_from_translated_context()

        return self.pun_list
