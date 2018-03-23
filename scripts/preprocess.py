import subprocess
import os
import sys
import json 
import tools
from tools.converters.conllu2sents import conllu2sents

def converter(config):
    data_types = config['data']['split'].keys()
    features = ['sents', 'gold_pos', 'gold_cpos', 'arcs', 'rels']
    for feature in features:
        for data_type in data_types:
            input_file = os.path.join(config['data']['base_dir'], config['data']['split'][data_type])
            output_file = os.path.join(config['data']['base_dir'], feature, data_type+'.txt')
            if not os.path.isdir(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            if feature == 'sents':
                index = 1
            elif feature == 'gold_pos':
                index = 4
            elif feature == 'gold_cpos':
                index = 3
            elif feature == 'rels':
                index = 7
            elif feature == 'arcs':
                index = 6
            conllu2sents(index, input_file, output_file)

def read_config(config_file):
    with open(config_file) as fhand:
        config_dict = json.load(fhand)
    return config_dict

######### main ##########

if __name__ == '__main__':
    config_file = sys.argv[1]
    config_file = read_config(config_file)
    print('Convert conllu+stag file to sentences, pos, stag, arcs, and rels')
    converter(config_file)
