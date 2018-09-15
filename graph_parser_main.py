from graph_parser_model import run_model, run_model_test
import os
from argparse import ArgumentParser 
import pickle
import sys

parser = ArgumentParser()
subparsers = parser.add_subparsers(title='different modes', dest = 'mode', description='train or test')
train_parser=subparsers.add_parser('train', help='train parsing')

# train options
train_parser.add_argument("--model", dest="model", help="model", default='Parsing_Model')
## data information
train_parser.add_argument("--base_dir", dest="base_dir", help="base directory for data")
train_parser.add_argument("--text_train", dest="text_train", help="text data for training")
train_parser.add_argument("--jk_train", dest="jk_train", help="jk data for training")
train_parser.add_argument("--jw_train", dest="jw_train", help="jw data for training")
train_parser.add_argument("--tag_train", dest="tag_train", help="tag data for training")
train_parser.add_argument("--arc_train", dest="arc_train", help="arc data for training")
train_parser.add_argument("--rel_train", dest="rel_train", help="rel data for training")
train_parser.add_argument("--text_test", dest="text_test", help="text data for testing")
train_parser.add_argument("--jk_test", dest="jk_test", help="jk data for testing")
train_parser.add_argument("--tag_test", dest="tag_test", help="tag data for testing")
train_parser.add_argument("--arc_test", dest="arc_test", help="arc data for testing")
train_parser.add_argument("--rel_test", dest="rel_test", help="rel data for testing")
train_parser.add_argument("--conllu_test", dest="conllu_test", help="conllu data for testing")

## model configuration

### LSTM config
train_parser.add_argument("--lstm", dest="lstm", help="rnn architecutre", type = int, default = 1)
train_parser.add_argument("--bidirectional", dest="bi", help="bidirectional LSTM", type = int, default = 1)
train_parser.add_argument("--max_epochs",  dest="max_epochs", help="max_epochs", type=int, default = 100)
train_parser.add_argument("--num_layers",  dest="num_layers", help="number of layers", type=int, default = 2)
train_parser.add_argument("--units", dest="units", help="hidden units size", type=int, default = 64)
train_parser.add_argument("--hidden_p", dest="hidden_p", help="keep fraction of hidden units", type=float, default = 1.0)
train_parser.add_argument("--dropout_p", dest="dropout_p", help="keep fraction", type=float, default = 1.0)
train_parser.add_argument("--word_embeddings_file", dest="word_embeddings_file", help="embeddings file", default = 'glovevector/glove.6B.100d.txt')
train_parser.add_argument("--elmo",  help="elmo", type=int, default=1)
train_parser.add_argument("--word_dropout", dest="word_dropout", help="keep fraction", type=float, default = 1.0)
train_parser.add_argument("--word_dropout_alpha", dest="word_dropout_alpha", help="keep fraction", type=float, default = -1.0)
train_parser.add_argument("--word_dropout_jw", dest="word_dropout_jw", help="keep fraction", type=float, default = 1.0)
## minus for not doing word dropout with frequency

### MLP config
train_parser.add_argument("--mlp_num_layers",  dest="mlp_num_layers", help="number of MLP layers", type=int, default = 1)
train_parser.add_argument("--arc_mlp_units", dest="arc_mlp_units", help="MLP units size", type=int, default = 64)
train_parser.add_argument("--rel_mlp_units", dest="rel_mlp_units", help="MLP units size", type=int, default = 32)
train_parser.add_argument("--joint_mlp_units", dest="joint_mlp_units", help="MLP units size", type=int, default = 500)
train_parser.add_argument("--mlp_prob", dest="mlp_prob", help="MLP units size", type=float, default = 1.0)

### Input Config
train_parser.add_argument("--stag_dim", dest="stag_dim", help="supertag dimension", type=int, default = 5)
train_parser.add_argument("--jk_dim", dest="jk_dim", help="jakcknife dimension", type=int, default = 5)
train_parser.add_argument("--embedding_dim", dest="embedding_dim", help="embedding dim", type=int, default = 100)
train_parser.add_argument("--early_stopping", dest="early_stopping", help="early stopping", type=int, default = 5)
train_parser.add_argument("--input_p", dest="input_dp", help="keep fraction for input", type=float, default = 1.0)

## Char Encoding
## Default same as Ma and Hovy 2016
train_parser.add_argument("--chars_dim", dest="chars_dim", help="character embedding dim", type=int, default = 30)
train_parser.add_argument("--chars_window_size", dest="chars_window_size", help="character embedding dim", type=int, default = 3)
train_parser.add_argument("--nb_filters", dest="nb_filters", help="nb_filters", type=int, default = 30)

### Train Config
train_parser.add_argument("--lrate", dest="lrate", help="lrate", type=float, default = 0.001)
train_parser.add_argument("--seed", dest="seed", help="set seed", type= int, default = 0)

### Scores
#train_parser.add_argument("--punc_test", dest="punc_test", help="punctuation data for testing")
train_parser.add_argument("--content_test", dest="content_test", help="content data for testing")
train_parser.add_argument("--metrics", nargs='+', dest="metrics", help="content data for testing")

## Output Options
train_parser.add_argument("--predicted_arcs_file", dest="predicted_arcs_file", help="filename for predicted arcs")
train_parser.add_argument("--predicted_rels_file", dest="predicted_rels_file", help="filename for predicted rels")
train_parser.add_argument("--predicted_conllu_file", dest="predicted_conllu_file", help="filename for predicted conllu")
#train_parser.add_argument("--predicted_arcs_file_greedy", dest="predicted_arcs_file_greedy", help="filename for predicted arcs")
#train_parser.add_argument("--predicted_rels_file_greedy", dest="predicted_rels_file_greedy", help="filename for predicted rels")
#train_parser.add_argument("--predicted_stags_file", dest="predicted_stags_file", help="filename for predicted rels") ## for joint
#train_parser.add_argument("--predicted_pos_file", dest="predicted_pos_file", help="filename for predicted rels") ## for joint

## test options
test_parser=subparsers.add_parser('test', help='test parser')
### data information
test_parser.add_argument("--base_dir", dest="base_dir", help="base directory for data")
test_parser.add_argument("--text_test", dest="text_test", help="text data for testing")
test_parser.add_argument("--jk_test", dest="jk_test", help="jk data for testing")
test_parser.add_argument("--tag_test", dest="tag_test", help="tag data for testing")
test_parser.add_argument("--arc_test", dest="arc_test", help="tag data for testing")
test_parser.add_argument("--rel_test", dest="rel_test", help="tag data for testing")
test_parser.add_argument("--conllu_test", dest="conllu_test", help="conllu data for testing")

### Model Information
test_parser.add_argument("--model", dest="modelname", help="model name")
### Output Options
test_parser.add_argument("--get_accuracy",  help="compute tag accuracy", action="store_true", default=False)
test_parser.add_argument("--save_tags", dest="save_tags", help="save 1-best tags")
test_parser.add_argument("--predicted_arcs_file", dest="predicted_arcs_file", help="filename for predicted arcs")
test_parser.add_argument("--predicted_rels_file", dest="predicted_rels_file", help="filename for predicted rels")
test_parser.add_argument("--predicted_conllu_file", dest="predicted_conllu_file", help="filename for predicted conllu")
#test_parser.add_argument("--predicted_arcs_file_greedy", dest="predicted_arcs_file_greedy", help="filename for predicted arcs")
#test_parser.add_argument("--predicted_rels_file_greedy", dest="predicted_rels_file_greedy", help="filename for predicted rels")
test_parser.add_argument("--predicted_stags_file", dest="predicted_stags_file", help="filename for predicted rels") ## for joint
test_parser.add_argument("--predicted_pos_file", dest="predicted_pos_file", help="filename for predicted rels") ## for joint
test_parser.add_argument("--save_probs", dest="save_probs", help="save probabilities", action="store_true", default=False)
test_parser.add_argument("--get_weight", dest="get_weight", help="save stag weight", action="store_true", default=False)

### Scores
#test_parser.add_argument("--punc_test", dest="punc_test", help="punctuation data for testing")
test_parser.add_argument("--content_test", dest="content_test", help="content data for testing")
test_parser.add_argument("--metrics", nargs='+', dest="metrics", help="content data for testing")

## Jabberwocky
test_parser.add_argument("--top_k", dest="top_k", help="Keep top 300", action="store_true", default=False)
test_parser.add_argument("--k", dest="k", help="Keep k", type=int, default=0)

opts = parser.parse_args()

if opts.mode == "train":
    print(opts.metrics)
#    opts.base_dir = 'sample_data'
#    opts.text_train = 'sample_data/sents/train.txt'
#    opts.tag_train = 'sample_data/predicted_stag/train.txt'
#    opts.jk_train = 'sample_data/predicted_pos/train.txt'
#    opts.arc_train = 'sample_data/arcs/train.txt'
#    opts.rel_train = 'sample_data/rels/train.txt'
#    opts.text_test = 'sample_data/sents/dev.txt'
#    opts.tag_test = 'sample_data/predicted_stag/dev.txt'
#    opts.jk_test = 'sample_data/predicted_pos/dev.txt'
#    opts.arc_test = 'sample_data/arcs/dev.txt'
#    opts.rel_test = 'sample_data/rels/dev.txt'
    params = ['bi', 'num_layers', 'units', 'hidden_p', 'dropout_p', 'mlp_num_layers', 'arc_mlp_units', 'rel_mlp_units', 'stag_dim', 'jk_dim', 'embedding_dim', 'input_dp', 'chars_dim', 'nb_filters', 'chars_window_size', 'lrate', 'seed', 'elmo']
    model_dir = '{}/'.format(opts.model) + '-'.join(map(lambda x: str(getattr(opts, x)), params))
    opts.model_dir = os.path.join(opts.base_dir, model_dir)
    if opts.word_dropout < 1.0:
        opts.model_dir += '-wd{}'.format(opts.word_dropout)
    if opts.word_dropout_alpha > 0.0:
        opts.model_dir += '-wa{}'.format(opts.word_dropout_alpha)
    if opts.word_dropout_jw < 1.0:
        opts.model_dir += '-wj{}'.format(opts.word_dropout_jw)
    #if 'glovevector' == opts.word_embeddings_file[:11]:
    #    opts.model_dir += '-glove'
    print('Model Dirctory: {}'.format(opts.model_dir))
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)
    with open(os.path.join(opts.model_dir, 'options.pkl'), 'wb') as fhand:
        pickle.dump(opts, fhand)
    opts.predicted_arcs_file = os.path.join(opts.model_dir, opts.predicted_arcs_file)
    if not os.path.isdir(os.path.dirname(opts.predicted_arcs_file)):
        os.makedirs(os.path.dirname(opts.predicted_arcs_file))
    opts.predicted_rels_file = os.path.join(opts.model_dir, opts.predicted_rels_file)
    if not os.path.isdir(os.path.dirname(opts.predicted_rels_file)):
        os.makedirs(os.path.dirname(opts.predicted_rels_file))
    opts.predicted_conllu_file = os.path.join(opts.model_dir, opts.predicted_conllu_file)
    if not os.path.isdir(os.path.dirname(opts.predicted_conllu_file)):
        os.makedirs(os.path.dirname(opts.predicted_conllu_file))
    run_model(opts)
    
if opts.mode == "test":
    with open(os.path.join(os.path.dirname(opts.modelname), 'options.pkl'), 'rb') as foptions:
        options=pickle.load(foptions)
    ## additional features not present in old models
    if not hasattr(options, 'word_dropout'):
        setattr(options, 'word_dropout', 1.0)
    if not hasattr(options, 'word_dropout_alpha'):
        setattr(options, 'word_dropout_alpha', 0.0)
    if not hasattr(options, 'word_dropout_jw'):
        setattr(options, 'word_dropout_jw', 1.0)
    run_model_test(options, opts)
