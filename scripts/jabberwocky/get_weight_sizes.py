import pickle, sys, os
import numpy as np

data_dir = sys.argv[1]
with open(os.path.join(data_dir, 'pos_weight.pkl'), 'rb') as fin:
    weights = pickle.load(fin)
with open(os.path.join(data_dir, 'pos_embeddings.pkl'), 'rb') as fin:
    pos_embeddings = pickle.load(fin)
pos_embeddings = pos_embeddings[:46]
with open(os.path.join(data_dir, 'word_embeddings.pkl'), 'rb') as fin:
    word_embeddings = pickle.load(fin)
word_embeddings = word_embeddings[1:51]
theta_x_g = weights['theta_x_g'][-25:]
theta_x_i = weights['theta_x_i'][-25:]
theta_x_g_word = weights['theta_x_g'][:100]
theta_x_i_word = weights['theta_x_i'][:100]
#print('POS Embeddings')
#print(np.linalg.norm(pos_embeddings))
#print('G Weight')
#print(np.linalg.norm(theta_x_g))
#print('I Weight')
#print(np.linalg.norm(theta_x_i))
#print('G Multiply')
#print(np.linalg.norm(pos_embeddings.dot(theta_x_g)))
#print('I Multiply')
#print(np.linalg.norm(pos_embeddings.dot(theta_x_i)))
#
#print('Bias')
#print(np.linalg.norm(weights['bias_g']))
#print(np.linalg.norm(weights['bias_i']))
print('Word Embeddings')
print(np.linalg.norm(word_embeddings))
print(np.linalg.norm(theta_x_i))
print(np.linalg.norm(theta_x_g))
#print('G Multiply')
#print(np.linalg.norm(word_embeddings.dot(theta_x_g_word)))
#print('I Multiply')
#print(np.linalg.norm(word_embeddings.dot(theta_x_i_word)))
