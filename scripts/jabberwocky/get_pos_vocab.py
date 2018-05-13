import numpy as np
import pickle
import sys
def process_glove(glove_file):
    words = []
    glove_vectors = []
    with open(glove_file) as fin:
        for line in fin:
            tokens = line.split()
            word = tokens[0]
            glove_vectors.append([float(x) for x in tokens[1:]])
            words.append(word)
    glove_vectors = np.array(glove_vectors)
    return words, glove_vectors
def process_trained(filename, word_count_file):
    words = []
    with open(filename, 'rb') as fin:
        word_embeddings = pickle.load(fin)
    word_embeddings = word_embeddings[1:-2]# padding; <-root->, unknown
    with open(word_count_file) as fin:
        for line in fin:
            tokens = line.split()
            words.append(tokens[0])
    return words, word_embeddings
def get_vocab(glove_words, words, glove_vectors, word_embeddings):
    vocab = [word for word in words if word in glove_words]
    with open('pos_vocab.txt', 'wt') as fout:
        for word in vocab:
            fout.write(word)
            fout.write('\n')
#def get_pos_data(trained_file, word_counts_file, glove_file):
#    word_embeddings = get_trained_file(trained_file)
#    glove_embeddings, word_list = get_glove_file(glove_file)
#
if __name__ == '__main__':
    glove_file = 'glovevector/glove.6B.100d.txt'
    word_count_file = 'scripts/jabberwocky/word_counts.txt'
    word_file = sys.argv[1]
    glove_words, glove_vectors = process_glove(glove_file)
    words, word_embeddings = process_trained(word_file, word_count_file)
    get_vocab(glove_words, words, glove_vectors, word_embeddings)
