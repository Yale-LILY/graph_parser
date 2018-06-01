#with open('data/conllu/en-ud-dev.conll16') as fhand:
import sys, pickle

#idx = 1 #word
def create_jabberwocky(conllu_file, words):
    output_file = 'test'
    with open(output_file, 'wt') as fout:
        with open(conllu_file) as fhand: 
            for line in fhand:
                tokens = line.split()
                if len(tokens) > 0:
                    if '#' not in tokens[0]: ## avoid comment lines
                        if tokens[1].lower() not in words:
                            tokens[1] = 'This_is_Jabberwocky'
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
