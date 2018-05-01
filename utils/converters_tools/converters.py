import os, io

def read_sents(sents_file):
    sents = []
    with io.open(sents_file, encoding='utf-8') as fhand:
        for line in fhand:
            sent = line.split()
            sents.append(sent)
    return sents

def output_conllu(test_opts):
    sents = read_sents(test_opts.text_test)
    #stags = read_sents(test_opts.predicted_stags_file)
    #pos = read_sents(test_opts.predicted_pos_file)
    arcs = read_sents(test_opts.predicted_arcs_file)
    rels = read_sents(test_opts.predicted_rels_file)
    #if not os.path.isdir(os.path.join(test_opts.base_dir, 'predicted_conllu')):
        #os.makedirs(os.path.join(test_opts.base_dir, 'predicted_conllu'))
    with io.open(test_opts.predicted_conllu_file, 'wt', encoding='utf-8') as fout:
        for sent_idx in range(len(sents)):
            sent = sents[sent_idx]
            #stags_sent = stags[sent_idx]
            #pos_sent = pos[sent_idx]
            arcs_sent = arcs[sent_idx]
            rels_sent = rels[sent_idx]
            for word_idx in range(len(sent)):
                line = [str(word_idx+1)]
                line.append(sent[word_idx]) ## already unicode
                line.append(u'_')
                line.append(u'_')
                line.append(u'_')
                #line.append(stags_sent[word_idx])
                #line.append(pos_sent[word_idx])
                line.append(u'_')
                line.append(str(arcs_sent[word_idx]))
                line.append(rels_sent[word_idx]) ## already unicode
                line.append(u'_')
                line.append(u'_')
                fout.write(u'\t'.join(line))
                fout.write(u'\n')
            fout.write(u'\n')

if __name__ == '__main__':
    print('running converter')
    class test_opts(object):
        def __init__(self):
            self.text_test = 'sample_data/sents/dev.txt'
            self.predicted_arcs_file = 'sample_data/arcs/dev.txt'
            self.predicted_rels_file = 'sample_data/rels/dev.txt'
            #self.predicted_arcs_file = 'dev_arcs.txt'
            #self.predicted_rels_file = 'dev_rels.txt'
            self.base_dir = 'sample_data'
            self.predicted_conllu_file = 'sample_data/conllu/dev.conllu'
    output_conllu(test_opts())
    
