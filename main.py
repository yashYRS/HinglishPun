import argparse
from pathlib import Path
from src.PunGenerator import PunGenerator
from src.homophone_gen import HomophoneGenerator
from src.utils import plot_survey_results


parser = argparse.ArgumentParser(description='')

parser.add_argument('-data_folder', type=str, default='data', help='Path to the Data folder')
parser.add_argument('-refine_folder', type=str, default='SelfRefine', help='Name of the folder where SelfRefine criteria is stored')
parser.add_argument('-prompt_folder', type=str, default='prompts', help='Name of the folder where prompts are stored')
parser.add_argument('-method', type=str, default='homophone_prompt', help='Method to be chosen for prompting')

parser.add_argument('-llm_model', type=str, default='gpt-3.5-turbo', help='Open AI model to choose for chat completion')
parser.add_argument('-llm_temp', type=float, default=0.3, help='Open AI LLM Temperature to choose randomness of text generation')

parser.add_argument('-load_homophone', dest='load_homophone', action='store_true', help='Whether homophones are to be generated from scratch')
parser.add_argument('-homophone_file_name', type=str, default='homophones.csv', help='Name of the file where the homophones are stored / need to be stored')

parser.add_argument('-log_puns', dest='log_puns', action='store_false', help='Print the puns to the terminal if not specified')
parser.add_argument('-plot_survey', dest='plot_survey', action='store_true', help='Plot the results of the survey')

parser.add_argument('-eval_transliterate', dest='eval_tranlisterate', action='store_true', help='Evaluate the transliteration method')
parser.add_argument('-save_trans_file', type=str, default='trans_eval.csv', help='transliteration evaluation results are saved in the given file')
parser.add_argument('-trans_file_name', type=str, default='aligned_dataset.tsv',
                    help='The dataset corpus tsv file which needs to be evaluated. Currently configured for Google Researchs Dakshina corpus')


args = parser.parse_args()

data_folder = Path(args.data_folder)
pun_method, load_homophone = args.method, args.load_homophone
homophone_df_path = data_folder / args.homophone_file_name
llm_model, llm_temp = args.llm_model, args.llm_temp

if args.plot_survey is True:
    survey_folder = data_folder / 'survey'     
    plot_survey_results(survey_folder)

elif args.eval_tranlisterate is True:
    file_name = args.trans_file_name
    file_path = data_folder / 'transliterate' / file_name
    save_path = data_folder / 'transliterate' / args.save_trans_file

    # Create the homophone generator object, to use the transliteration module
    hom_gen = HomophoneGenerator(data_folder=data_folder)

    # Evaluate the transliteration object
    accuracy = hom_gen.evaluate_transliteration(file_path, save_file=save_path)
    print(" Transliteration accuracy on given datastet - {}".format(accuracy))

else:
    # Create the Pun Generation object
    pun_gen_obj = PunGenerator(data_folder=data_folder, 
                               homophone_df_path=homophone_df_path,
                               load_homophone=load_homophone,
                               method=pun_method,
                               llm_model_name=llm_model,
                               llm_temperature=llm_temp)

    # Get puns using the method suggested, if logging = True, prints puns to the terminal
    # Else it gets saved to the prompts/../results folders

    self_refine_folder = None
    if pun_method == 'homophone_prompt':
        prompt_folder = Path(args.prompt_folder) / 'input_homophone'
        if args.refine_folder is not None:
            self_refine_folder = Path(args.prompt_folder) / args.refine_folder
    elif pun_method == 'prompt':
        prompt_folder = Path(args.prompt_folder) / 'algorithmic'
    else:
        prompt_folder = Path(args.prompt_folder)

    pun_list = pun_gen_obj.get_puns(self_refine=self_refine_folder, prompt_folder=prompt_folder, log_puns=args.log_puns)
