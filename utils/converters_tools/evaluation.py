#!/usr/bin/env python3
import subprocess
import os

def get_scores(test_opts):
    command = 'python evaluation_script/conll17_ud_eval.py'
    ## add gold_file
    command = command + ' ' + test_opts.conllu_test
    ## add system_file
    command = command + ' ' + test_opts.predicted_conllu_file
    ## verbose
    command += ' -v'
    ## add weight
    command += ' --weight evaluation_script/weights.clas'
    output = subprocess.check_output(command, shell=True)
    lines = output.decode('utf-8').split('\n')
    scores = {}
    for i, line in enumerate(lines):
        items = line.split('|')
        if i >= 2 and len(items) >= 4:
            scores[items[0].strip()] = float(items[3].strip())
    return scores

if __name__ == '__main__':
    class test_opts(object):
        def __init__(self):
            self.text_test = 'sample_data/sents/dev.txt'
            self.predicted_arcs_file = 'sample_data/arcs/dev.txt'
            self.predicted_rels_file = 'sample_data/rels/dev.txt'
            #self.predicted_arcs_file = 'dev_arcs.txt'
            #self.predicted_rels_file = 'dev_rels.txt'
            self.base_dir = 'sample_data'
            self.predicted_conllu_file = 'sample_data/predicted_conllu/dev.conllu'
            self.conllu_test = 'sample_data/conllu/dev.conllu'
    scores = get_scores(test_opts())
    print(scores)
