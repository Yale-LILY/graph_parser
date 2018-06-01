from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle, sys, random, itertools
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    new_embeddings = new_embeddings[idxes]
    return new_embeddings

def get_data(path_to_vocab, path_to_pos_y, path_to_word_embeddings, path_to_word_embeddings_vocab):
    idxes = get_y(path_to_pos_y) 
    words = get_vocab(path_to_vocab)
    #new_glove = get_glove(path_to_glove, path_to_glove_vocab, words)
    new_embeddings = get_new_embeddings(path_to_word_embeddings, path_to_word_embeddings_vocab, words)
    return new_embeddings, idxes

def regress(x, y):
    confusion = np.zeros([10, 10])
    random.seed(0)
    nb_words = x.shape[0]
    perm = np.arange(nb_words)
    random.shuffle(perm)
    k_fold = 5
    x = x[perm]
    y = y[perm]
    x_groups = np.split(x, k_fold)
    y_groups = np.split(y, k_fold)
    nb_hold_out = nb_words/k_fold
    for i_th in range(k_fold):
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)
        x_test = x_groups[i_th]
        y_test = y_groups[i_th]
        x_train = np.vstack(x_groups[:i_th] + x_groups[i_th+1:])
        y_train = np.hstack(y_groups[:i_th] + y_groups[i_th+1:])
    # all parameters not specified are set to their defaults
        print(x_test.shape)
        print(y_test.shape)
        print(x_train.shape)
        print(y_train.shape)
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
        #print(x_test.shape)
        #print(y_test.shape)
        #print(x_train.shape)
        #print(y_train.shape)
        logisticRegr = LogisticRegression()
        logisticRegr.fit(x_train, y_train)
        y_predict = logisticRegr.predict(x_test)
        confusion += confusion_matrix(y_test, y_predict)
    #confusion = confusion/np.sum(confusion, axis=1)
    return confusion

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('test')



if __name__ == '__main__':
    path_to_vocab = 'scripts/jabberwocky/pos_vocab_noties_top10.txt'
    path_to_pos_y = 'scripts/jabberwocky/pos_y_noties_top10.txt'  
    path_to_word_embeddings = sys.argv[1]
    path_to_word_embeddings_vocab = 'scripts/jabberwocky/word_counts.txt' 
    new_embeddings, idxes = get_data(path_to_vocab, path_to_pos_y, path_to_word_embeddings, path_to_word_embeddings_vocab)
    confusion = regress(new_embeddings, idxes)

    class_names = ['CD', 'IN', 'JJ', 'NN', 'NNP', 'NNS', 'RB', 'VB', 'VBD', 'VBN']
    np.set_printoptions(precision=2)
    plot_confusion_matrix(confusion, classes=class_names,
			  title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
