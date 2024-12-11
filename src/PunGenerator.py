import os
import ast
import tqdm

import logging
import random
from pathlib import Path
from thefuzz import fuzz

from dotenv import load_dotenv

from nltk import pos_tag
from openai import OpenAI

from gensim import downloader as api

from src import utils
from src.homophone_gen import HomophoneGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer


logging.getLogger().setLevel(logging.INFO)

class OpenSourceLLMClient:
    def __init__(self, model_name, temperature=0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.temperature = temperature

    def generate_response(self, message_prompt):
        inputs = self.tokenizer(message_prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            max_length=400
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


class PunGenerator:

    def __init__(self, data_folder: Path, homophone_df_path: Path, load_homophone: bool=False, method: str='homophone',
                 llm_model_name: str="gpt-3.5-turbo", llm_temperature: float=0.3, limit_calls: int=1, refine_iterations: int=3):
        """
        Args:
            data_folder (Path): Path to the folder containing resources like the transliterated character files, common
                English words, hindi words etc.
            homophone_df_path (Path): Path to the file where the homophone df is either stored or will be stored
            load_homophone (bool, optional): If False, homophones are generated from scratch. Defaults to False.
            method (str, optional): Possible types are `prompting`, `homophone`, `homophone_prompt`. 
                which refer to standard instruction following prompting where the model is supposed to find the
                homophone itself as well, where we generate the homophones and even the subsequent joke without the model
                through GLOve embeddings and finally where we provide the LLM with homophones and ask it to generate
                puns by exploiting them. Defaults to 'homophone'.
            llm_model_name (str, optional): The Open AI model of choice. Defaults to "gpt-3.5-turbo".
            llm_temperature (float, optional): Lower temperature results in more predictable responses. Defaults to 0.3.
            limit_calls (int, optional): Number of homophones to pass per Prompt type. Defaults to 1
            refine_iterations (int, optional): Number of tries for SelfRefine framework.            
        """
        if method == 'homophone':
            # The number of possible replacements that will be used to replace the existing noun phrase
            self.top_n_context_thresh: int = 1
            self.noun_pos_tags: set = set(['NN', 'NNS', 'NNP', 'NNPS'])

            # Load pre-trained Word2Vec model
            self.embedding_model = api.load("glove-twitter-25")
            # Store the entire vocabulary found in the embedding model to allow faster membership inference
            self.embedding_vocab : set = set(self.embedding_model.index_to_key)
            self.setup_homophone_df(data_folder, load_homophone, homophone_df_path, generate_sentence=True)

        elif method == 'prompt':
            self.setup_llm_client(llm_model_name, llm_temperature)
        
        elif method == 'homophone_prompt':
            self.limit_calls = limit_calls
            self.setup_homophone_df(data_folder, load_homophone, homophone_df_path)
            self.setup_llm_client(llm_model_name, llm_temperature)

        self.open_source = False
        self.refine_iterations: int = 3
        self.method: str = method 
        self.pun_list: list = []
    
    def setup_homophone_df(self, data_folder: Path, load_homophone: bool, homophone_df_path: Path, generate_sentence: bool=False):
        """Either load the homophone dataframe or create it from scratch by comparing IPAs of English and Hindi words

        Args:
            data_folder (Path): Path to the folder containing resources like the transliterated character files, common
                English words, hindi words etc.
            load_homophone (bool): If False, homophones are generated from scratch
            homophone_df_path (Path): Path to the file where the homophone df is either stored or will be stored
            generate_sentence (bool, optional):If True, then candidate sentences per English word in our dataframe 
                will be also be stored in our dataframe to aid in constructing puns. Defaults to False.
        """        
        if load_homophone is True:
            # If homophones have already been generated, load the file 
            self.homophone_gen = HomophoneGenerator(data_folder=data_folder, load_df_file=homophone_df_path)
            homophone_df = self.homophone_gen.homophone_df

            # Since we are loading the dataframe from a csv, interpret string representation of lists as lists
            if 'candidate_sentences' in homophone_df.columns:
                homophone_df['candidate_sentences'] = homophone_df['candidate_sentences'].apply(ast.literal_eval)

            homophone_df['dvng_hi'] = homophone_df['dvng_hi'].apply(ast.literal_eval)
            homophone_df['latin_hi'] = homophone_df['latin_hi'].apply(ast.literal_eval)
            homophone_df['translated_hi_en'] = homophone_df['translated_hi_en'].apply(ast.literal_eval)
        else:
            # Create the homophone df from scratch
            self.homophone_gen = HomophoneGenerator(data_folder=data_folder, save_df_file=homophone_df_path)
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
        if 'gpt' in llm_model_name: 
            api_key_value = os.getenv("OPENAPI_KEY")
            # Initialise the OPEN AI model
            self.llm_client = OpenAI(api_key=api_key_value)
        elif 'llama' in llm_model_name.lower():
            api_key_value = os.getenv("LLAMA_KEY")
            os.environ["HF_ACCESS_TOKEN"] = api_key_value
            self.llm_client = OpenSourceLLMClient(llm_model_name, llm_temperature)
            self.open_source = True
        else:
            logging.info("LLM model not supported")
            exit(0)

    def replace_noun_for_pun(self, input_sentence: list, pos_sequence: list, word_context_change: str, log_puns: bool=True):
        """Replace the first noun phrase of the sentence with a corresponding noun phrase that is more
        related to the context of the given input word 

        Args:
            input_sentence (list): Input sentence which needs to be changed
            pos_sequence (list): Pos tagged data of the given sentence 
            word_context_change (str): Word based on which, the starting noun phrase will be changed
            log_puns (bool, optional): If True Print the generated puns to the terminal. Defaults to True            
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
                    if log_puns is True:
                        logging.info('Pun - {} '.format(' '.join(new_sentence)))
                    # Add the generated sentence to the pun list
                    self.pun_list.append(' '.join(new_sentence))

    def get_pun_from_translated_context(self, log_puns: bool=True):
        """For all the homophones in the dataframe, try and generate puns by manipulating some 
        sentences corresponding to the English word. Many sentences where the english word from the 
        homophone pair lies at the end of the sentence. In such cases, the English word is replaced by its
        homphonic hindi pair, and the starting Noun phrase of the sentence is changed to make the overall
        sentence still make sense.
            log_puns (bool, optional): If True Print the generated puns to the terminal. Defaults to True         
        """
        for _, row in tqdm.tqdm(self.homophone_gen.homophone_df.iterrows()):
            # For all the sentences where the row 'en' word appears at the end, try to create puns
            # by replace the noun at the start with a potential context word of the associated homophone word
            for input_sentence in row.candidate_sentences:
                # If the sentence is too long, skip
                if len(input_sentence) > 20:
                    continue
                            
                for hi_latin, hi_word, translate_word in zip(row.latin_hi, row.dvng_hi, row.translated_hi_en):
                    if fuzz.ratio(translate_word.lower(), hi_latin) > 50:
                        # The hi word is most probably a borrowed word, hence skip
                        continue
                    # If the meanings of the homophones are the same, don't create the pun
                    if translate_word.lower() == row.en:
                        continue
                    
                    # The transliterated word should be of similar length to the english word
                    # The homophonic word shouldn't mean the same thing in Hindi as it does in English                    
                    if abs(len(row.en) - len(hi_latin)) < 2 and fuzz.ratio(translate_word.lower(), row.en) < 85:
                        continue

                    # Replace the English word of the homophonic pair with the Hindi word 
                    new_sentence = [i if i != row.en else hi_word for i in input_sentence]
                    # Tag the sentence to identify the noun phrase at the start
                    pos_sequence = pos_tag(new_sentence)
                    # Replace the Noun Phrase at the start with a more contextual noun
                    self.replace_noun_for_pun(new_sentence, pos_sequence, translate_word, log_puns)

                # To limit the number of puns being generated
                break
    
    def execute_llm_call(self, message_prompt):
        """Call the LLM OPENAI client with the given message prompt

        Args:
            message_prompt (str): the Prompt to be passed to the LLM

        Returns:
            str: Response from the LLM
        """        
        # Call the LLM Api to get the required response
        # Temperature is high, since we want more deterministic outputs
        if self.open_source is False:
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
        else: 
            # Open source LLM client uses transformer generate method
            self.llm_client.generate_response(message_prompt)
        return response

    def prompting_llm_algorithmic(self, prompt_folder: Path=None, log_puns: bool=True):
        """Prompt LLMs by describing a basic algorithm for generating puns

        Args:
            prompt_folder (Path, optional): Path where the prompts are stored. Defaults to None.
            log_puns (bool, optional): If True Print the generated puns to the terminal. Defaults to True 
        """        
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
                if log_puns is True:
                    logging.info("Method - {} pun  - {} ".format(prompt_type_dir.name, required_pun))
                # Add the generated pun to the list of puns generated by the system
                self.pun_list.append(required_pun)
                
                if result_file.exists():
                    # Append the new prompt result to the result file
                    existing_data = utils.read_file(result_file)
                    response = existing_data + '\n' + response

                # Write the raw response received from the LLM to the result files
                utils.write_file(response, result_file)

    def prompting_llm_homophone(self, self_refine: Path=None, prompt_folder: Path=None, log_puns: bool=True):
        """Prompt LLMs by providing them Homophone pairs as input along with some examples of 
        the kind of puns / one liners we are looking for, given such a homophonic pair

        Args:
            self_refine (Path, optional): Path to where the Self refine paths are stored.  
            prompt_folder (Path, optional): Path where the prompts are stored. Defaults to None.
            log_puns (bool, optional): If True Print the generated puns to the terminal. Defaults to True            
        """
        prompt_instruct_folder = prompt_folder / 'input'
        
        # Zero shot / One shot / Few shot
        if self_refine is not None:
            refine_zero_shot = utils.read_file(self_refine / 'zero_shot.txt')
            refine_few_shot = utils.read_file(self_refine / 'few_shot.txt')
            prompt_result_folder = prompt_folder / 'refine_results'
        else: 
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
            homophone_strings = random.sample(homophone_strings, self.limit_calls)
            
            for homophone_inp in homophone_strings:
                input_prompt = message_prompt + '\n' + homophone_inp
                
                # Call the LLM model
                if self_refine is None:
                    response = self.execute_llm_call(input_prompt)
                elif '0' in input_file.stem:
                    response = self.self_refine(prompt=input_prompt, criteria=refine_zero_shot)
                else: 
                    response = self.self_refine(prompt=input_prompt, criteria=refine_few_shot)
                # Remove the extra details and just keep the pun
                required_pun = utils.post_process_llm_response(response)
                # # Add the generated pun to the list of puns generated by the system
                self.pun_list.append(required_pun)
                
                # Add the homophone given as input during the prompt
                response = homophone_inp + '\n' + response
                if log_puns is True:
                    logging.info("Method - {} pun  - {} ".format(input_file.stem, response))

                if result_file.exists():
                    # Append the new prompt result to the result file
                    existing_data = utils.read_file(result_file)
                    response = existing_data + '\n' + response

                # Write the raw response received from the LLM to the result files
                utils.write_file(response, result_file)
    
    def get_puns(self, self_refine: Path=None, prompt_folder: Path=None, log_puns: bool=True) -> list:
        """Wrapper function to provide a unified API for generating puns from this package

        Args:
            self_refine (Path, optional): Path to where the Self Refine path is stored.
                If none, don't use self refine. Defaults to None.
            prompt_folder (Path, optional): Path to where the prompts are stored. Defaults to None.
                Doesn't need to be passed in case method doesn't involve prompting
            log_puns (bool, optional): If True Print the generated puns to the terminal. Defaults to True

        Returns:
            list: List of puns generated
        """
        if self.method == 'prompt':
            # Ask the LLM to follow a algorithm to generate puns
            self.prompting_llm_algorithmic(prompt_folder, log_puns)
        
        elif self.method == 'homophone_prompt':
            # Input homophones, ask LLM to generate puns based on them.
            self.prompting_llm_homophone(self_refine, prompt_folder, log_puns)

        elif self.method == 'homophone':
            # LLM isn't involved, generate purely algorithmically
            self.get_pun_from_translated_context(log_puns)

        return self.pun_list


    def self_refine(self, prompt, criteria):
        """
        Use SelfRefine to improve an initial GPT response based on iterative self-assessment.
        
        Parameters:
        - prompt (str): The initial prompt to generate the response.
        - criteria (str): Criteria for self-assessment.
        - iterations (int): Number of refinement iterations.

        Returns:
        - Final refined response.
        """
        # Step 1: Generate the initial response
        response = self.execute_llm_call(prompt)
        logging.info('NEWWW Response - {} '.format(response))
        # Iterative refinement
        for i in range(self.refine_iterations):
            # Step 2: Self-assessment
            assessment_prompt = f"""
            Review the Output based on these criteria:
            {criteria}
            
            Output:
            {response}
            
            List issues for each criterion and suggest improvements.
            """
            assessment = self.execute_llm_call(assessment_prompt)
            logging.info('Assessment - {} '.format(assessment))
            # Step 3: Refinement
            refinement_prompt = f"""
            Based on the Assessment, refine the output to address identified issues.
            
            Assessment:
            {assessment}
            
            Current Output:
            {response}
            
            Provide a refined version.
            """
            refined_response = self.execute_llm_call(refinement_prompt)
            logging.info('Response - {} '.format(response))
            # Update the response for the next iteration
            response = refined_response
        
        return response

