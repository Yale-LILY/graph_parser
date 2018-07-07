import subprocess
import os
import sys
import json 

def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

def train_parser(config):
    base_dir = config['data']['base_dir']
    model_type = config['parser']['model_options']['model']
    features = ['sents', 'gold_pos', 'gold_pos', 'arcs', 'rels']
    #print('Use UPOS')
    #features = ['sents', 'gold_cpos', 'gold_cpos', 'arcs', 'rels']
    base_command = 'python graph_parser_main.py train --base_dir {}'.format(base_dir)
    train_data_dirs = map(lambda x: os.path.join(base_dir, x, 'train.txt'), features)
    train_data_info = ' --text_train {} --jk_train {} --tag_train {} --arc_train {} --rel_train {}'.format(*train_data_dirs)
    #train_data_info = ' --jw_train {}'.format(os.path.join(base_dir, 'syntactic_jw', 'train.txt'))
    dev_data_dirs = map(lambda x: os.path.join(base_dir, x, 'dev.txt'), features)
    dev_data_info = ' --text_test {} --jk_test {} --tag_test {} --arc_test {} --rel_test {}'.format(*dev_data_dirs)
    dev_data_info += ' --conllu_test {}'.format(os.path.join(base_dir, 'conllu', 'dev.conllu'))
    model_config_dict = config['parser']
    model_config_info = ''
    for param_type in model_config_dict.keys():
        for option, value in model_config_dict[param_type].items():
            model_config_info += ' --{} {}'.format(option, value)
    complete_command = base_command + train_data_info + dev_data_info + model_config_info
    #complete_command += ' --max_epochs 1' ## for debugging
    output_info = ''
    output_file = os.path.join('predicted_arcs', '{}.txt'.format('dev'))
    #if not os.path.isdir(os.path.dirname(output_file)):
    #    os.makedirs(os.path.dirname(output_file))
    output_info += ' --predicted_arcs_file {}'.format(output_file)
    output_file = os.path.join('predicted_rels', '{}.txt'.format('dev'))
    #if not os.path.isdir(os.path.dirname(output_file)):
    #    os.makedirs(os.path.dirname(output_file))
    output_info += ' --predicted_rels_file {}'.format(output_file)
    output_file = os.path.join('predicted_conllu', '{}.conllu'.format('dev'))
    #if not os.path.isdir(os.path.dirname(output_file)):
    #    os.makedirs(os.path.dirname(output_file))
    output_info += ' --predicted_conllu_file {}'.format(output_file)
    complete_command += output_info
    subprocess.check_call(complete_command, shell=True)

######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
    print('Train Parser')
    train_parser(config_file)
