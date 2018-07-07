#with open('data/conllu/en-ud-dev.conll16') as fhand:
import sys, pickle
import numpy as np

#idx = 1 #word
np.random.seed(0)
pos2word_dict = {}
with open('scripts/jabberwocky/pos2word.txt') as fin:
    for line in fin:
        pos = line.split()[0]
        words = line.split()[1:]
        pos2word_dict[pos] = words
    
def pos2word(pos):
    if pos in pos2word_dict:
        words = pos2word_dict[pos]
        i = np.random.randint(0, len(words))
        return words[i]
    else:
        return 'This_is_Jabberwocky'
    
def create_jabberwocky(conllu_file, words):
    output_file = 'test'
    with open(output_file, 'wt') as fout:
        with open(conllu_file) as fhand: 
            for line in fhand:
                tokens = line.split()
                if len(tokens) > 0:
                    if '#' not in tokens[0]: ## avoid comment lines
                        lemma = tokens[2]
                        UPOS = tokens[3]
                        XPOS = tokens[4]
                        if UPOS in ['NOUN', 'PROPN', 'NUM', 'ADJ', 'ADV']:
                            tokens[1] = pos2word(XPOS)
                        elif UPOS in ['VERB']:
                            if lemma not in ['be', 'have', 'do'] and XPOS not in ['MD']:
                                tokens[1] = pos2word(XPOS)
                        fout.write('\t'.join(tokens))
                        fout.write('\n')
                else:
                    fout.write('\n')

    #with open('dev.txt', 'wt') as fhand:
#idx = 4 #pos
#idx = 10 #stag
#idx =  7#relation
#idx = 6 #parent
def get_topk(filename, k):
    words = []
    with open(filename) as fin:
        for line in fin:
            word = line.split()[0]
            words.append(word)
            if len(words) == k:
                return words
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit('')
    conllu_file = sys.argv[1]

#    conllu2sents(idx, 'data/conllu/en-ud-train_parsey.txt', 'train.txt') 
    #create_jabberwocky(conllu_file)
    words = get_topk('scripts/jabberwocky/word_counts.txt', 100)
    create_jabberwocky(conllu_file, words)
#    conllu2sents(4, 'data/conllu/en-ud-dev_parsey.txt', 'dev.txt') 
#    conllu2sents(4, 'dev', 'dev.txt') 
#    conllu2sents(4, 'train_long', 'train.txt') 
