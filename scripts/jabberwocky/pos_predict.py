from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle, sys

def get_vocab(path_to_vocab):
    words = []
    with open(path_to_vocab) as fin:
        for line in fin:
            word = line.split()[0]
            words.append(word)
    return words
def get_y(path_to_pos_y):
    idxes = []
    with open(path_to_pos_y) as fin:
        for line in fin:
            pos, idx = line.split()
            idxes.append(int(idx))
    idxes = np.array(idxes)
    return idxes
def get_glove(path_to_glove, path_to_glove_vocab, words):
    word_idx = 0
    glove_words = get_vocab(path_to_glove_vocab)
    idxes = []
    for idx, glove_word in enumerate(glove_words):
        if glove_word == words[word_idx]:
            idxes.append(idx)
            word_idx += 1
            if len(words) == word_idx:
                break
    idxes = np.array(idxes)
    with open(path_to_glove, 'rb') as fin:
        pos_glove = pickle.load(fin)
    print(len(glove_words), pos_glove.shape[0])
    new_glove = pos_glove[idxes]
    return new_glove

def get_new_embeddings(path_to_word_embeddings, path_to_word_embeddings_vocab, words):
    word_idx = 0
    embedding_words = get_vocab(path_to_word_embeddings_vocab)
    idxes = []
    for idx, embedding_word in enumerate(embedding_words):
        if embedding_word == words[word_idx]:
            idxes.append(idx)
            word_idx += 1
            if len(words) == word_idx:
                break
    idxes = np.array(idxes)
    with open(path_to_word_embeddings, 'rb') as fin:
        new_embeddings = pickle.load(fin)
    new_embeddings = new_embeddings[1:-2] #skip padding, -root-, and -unk-
    print(len(embedding_words), new_embeddings.shape[0])
    new_embeddings = new_embeddings[idxes]
    return new_embeddings

def get_data(path_to_vocab, path_to_pos_y, path_to_glove, path_to_glove_vocab, path_to_word_embeddings, path_to_word_embeddings_vocab):
    idxes = get_y(path_to_pos_y) 
    words = get_vocab(path_to_vocab)
    new_glove = get_glove(path_to_glove, path_to_glove_vocab, words)
    new_embeddings = get_new_embeddings(path_to_word_embeddings, path_to_word_embeddings_vocab, words)
    return new_glove, new_embeddings, idxes

def regress(x, y):
    nb_seeds = 5
    scores = np.zeros([nb_seeds])
    for seed in range(nb_seeds):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)
    # all parameters not specified are set to their defaults
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        score = logisticRegr.score(x_test, y_test)
        scores[seed] = score
    return np.mean(scores)

#

if __name__ == '__main__':
    path_to_vocab = 'scripts/jabberwocky/pos_vocab_noties_top10.txt'
    path_to_pos_y = 'scripts/jabberwocky/pos_y_noties_top10.txt'  
    path_to_glove = 'scripts/jabberwocky/pos_glove.pkl' 
    path_to_glove_vocab = 'scripts/jabberwocky/pos_vocab.txt' 
    #path_to_word_embeddings = '/data/lily/jk964/models/WSJ/Parsing_Model/1-3-400-0.67-0.67-1-500-100-0-25-100-0.67-0-0-3-0.001-0-wd0.2/word_embeddings.pkl'
    path_to_word_embeddings = sys.argv[1]
    path_to_word_embeddings_vocab = 'scripts/jabberwocky/word_counts.txt' 
    new_glove, new_embeddings, idxes = get_data(path_to_vocab, path_to_pos_y, path_to_glove, path_to_glove_vocab, path_to_word_embeddings, path_to_word_embeddings_vocab)
    print(new_glove.shape)
    print(new_embeddings.shape)
    mean_score = regress(new_glove, idxes)
    print(round(mean_score*100, 2))
    mean_score = regress(new_embeddings, idxes)
    print(round(mean_score*100, 2))

#digits = load_digits()
## Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)print(“Image Data Shape”, digits.data.shape)
## Print to show there are 1797 labels (integers from 0–9)
#print("Label Data Shape", digits.target.shape)
#print(digits.data.shape)
#print(digits.target.shape)
#print(digits.data[:10])
#print(digits.target[:10])
#x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
#
## all parameters not specified are set to their defaults
#logisticRegr = LogisticRegression()
#logisticRegr.fit(x_train, y_train)
#score = logisticRegr.score(x_test, y_test)
#print(score)
