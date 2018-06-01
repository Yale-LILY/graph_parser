#with open('data/conllu/en-ud-dev.conll16') as fhand:
import sys

#idx = 1 #word
def create_jabberwocky(conllu_file):
    output_file = 'test'
    unique_keep_words = []
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
                            tokens[1] = 'This_is_Jabberwocky'
                        elif UPOS in ['VERB']:
                            if lemma not in ['be', 'have', 'do'] and XPOS not in ['MD']:
                                tokens[1] = 'This_is_Jabberwocky'
                        if tokens[1] != 'This_is_Jabberwocky':
                            keep_word = tokens[1]
                            #if keep_word not in unique_keep_words:
                            #    unique_keep_words.append(keep_word)
                        fout.write('\t'.join(tokens))
                        fout.write('\n')
                else:
                    fout.write('\n')
    #print(unique_keep_words)
    print(len(unique_keep_words))

    #with open('dev.txt', 'wt') as fhand:
#idx = 4 #pos
#idx = 10 #stag
#idx =  7#relation
#idx = 6 #parent
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit('')
    conllu_file = sys.argv[1]

#    conllu2sents(idx, 'data/conllu/en-ud-train_parsey.txt', 'train.txt') 
    create_jabberwocky(conllu_file) 
#    conllu2sents(4, 'data/conllu/en-ud-dev_parsey.txt', 'dev.txt') 
#    conllu2sents(4, 'dev', 'dev.txt') 
#    conllu2sents(4, 'train_long', 'train.txt') 
