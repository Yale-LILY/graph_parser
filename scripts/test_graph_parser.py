import subprocess
import os
import sys
import json 
import tools
from argparse import ArgumentParser 

parser = ArgumentParser()
parser.add_argument('config_file', metavar='N', help='an integer for the accumulator')
parser.add_argument('model_name', metavar='N', help='an integer for the accumulator')
parser.add_argument("--no_gold",  help="compute tag accuracy", action="store_true", default=False)
parser.add_argument("--save_probs",  help="save tag probabilities", action="store_true", default=False)
parser.add_argument("--get_weight",  help="get stag weight", action="store_true", default=False)
opts = parser.parse_args()


def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

def test_parser(config, best_model, data_types, no_gold):
    base_dir = config['data']['base_dir'] 
    model_type = config['parser']['model_options']['model']
    if model_type == 'Parsing_Model_Joint':
        print('Run joint training. Use gold supertags')
        if no_gold:
            features = ['sents', 'predicted_pos', 'sents', 'sents', 'sents']
        else:
            features = ['sents', 'predicted_pos', 'gold_stag', 'arcs', 'rels']
    elif model_type == 'Parsing_Model_Joint_Both':
        print('Run joint training. Use gold supertags')
        if no_gold:
            features = ['sents', 'sents', 'sents', 'sents', 'sents']
        else:
            features = ['sents', 'gold_pos', 'gold_stag', 'arcs', 'rels']
    else:
        if no_gold:
            features = ['sents', 'predicted_pos', 'predicted_stag', 'sents', 'sents']
        else:
            features = ['sents', 'predicted_pos', 'predicted_stag', 'arcs', 'rels']

    if no_gold:
        base_command = 'python graph_parser_main.py test'
    else:
        base_command = 'python graph_parser_main.py test --get_accuracy'
    model_info = ' --model {}'.format(best_model)
    for data_type in data_types:
        base_command += ' --conllu_test {}.conllu'.format(os.path.join(base_dir, 'conllu', data_type))
        if model_type in ['Parsing_Model_Joint', 'Parsing_Model_Joint_Both']:
            output_file = os.path.join(base_dir, 'predicted_stag', '{}.txt'.format(data_type))
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            output_info = ' --predicted_stags_file {}'.format(output_file)
            output_file = os.path.join(base_dir, 'predicted_pos', '{}.txt'.format(data_type))
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            output_info += ' --predicted_pos_file {}'.format(output_file)
        else:
            output_info = '' ## no stag output
        output_file = os.path.join(base_dir, 'predicted_arcs', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_arcs_file {}'.format(output_file)
        output_file = os.path.join(base_dir, 'predicted_rels', '{}.txt'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_rels_file {}'.format(output_file)
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_file = os.path.join(base_dir, 'predicted_conllu', '{}.conllu'.format(data_type))
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        output_info += ' --predicted_conllu_file {}'.format(output_file)
        model_config_dict = config['parser']
        for param_type in model_config_dict.keys():
            if param_type == 'scores':
                for option, value in model_config_dict[param_type].items():
                    model_config_info = ' --{} {}'.format(option, value)
        if opts.save_probs: 
            output_info += ' --save_probs'
        if opts.get_weight: 
            output_info += ' --get_weight'
        if no_gold:
            test_data_dirs = map(lambda x: os.path.join(base_dir, x, '{}.txt'.format(data_type)), features)
            test_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {}'.format(*test_data_dirs)
        else:
            test_data_dirs = map(lambda x: os.path.join(base_dir, x, '{}.txt'.format(data_type)), features)
            test_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {}'.format(*test_data_dirs)
        complete_command = base_command + model_info + output_info + test_data_info + model_config_info
        subprocess.check_call(complete_command, shell=True)
######### main ##########

if __name__ == '__main__':
    config_file = opts.config_file
    config_file = read_config(config_file)
    best_model = opts.model_name
    data_types = config_file['data']['split'].keys()
    #data_types = [x for x in data_types if x!='train']
    data_types = ['dev']
    test_parser(config_file, best_model, data_types, opts.no_gold)
