import numpy as np
import pickle
import sys
def process_glove(glove_file, vocab_file):
    words = []
    with open(vocab_file) as fin:
        for line in fin:
            word = line.split()[0]
            words.append(word)
    glove_vectors = []
    word2glove = {}
    with open(glove_file) as fin:
        for line in fin:
            tokens = line.split()
            word = tokens[0]
            word2glove[word] = np.array([float(x) for x in tokens[1:]])
            
    for word in words:
        glove_vectors.append(word2glove[word])
    glove_vectors = np.array(glove_vectors)
    with open('scripts/jabberwocky/pos_glove.pkl', 'wb') as fout:
        pickle.dump(glove_vectors, fout)
    
if __name__ == '__main__':
    glove_file = 'glovevector/glove.6B.100d.txt'
    vocab_file = 'scripts/jabberwocky/pos_vocab.txt'
    process_glove(glove_file, vocab_file)
