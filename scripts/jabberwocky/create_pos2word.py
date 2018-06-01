from collections import defaultdict
def read_file(filename, lower=False):
    tokens = []
    with open(filename) as fin:
        for line in fin:
            if lower:
                line = line.lower()
            tokens.extend(line.split())
    return tokens
def get_count(words, pos_tags):
    count = defaultdict(lambda: defaultdict(int))
    for word, pos_tag in zip(words, pos_tags):
        count[pos_tag][word] += 1
    return count

def get_vocab(vocab_file):
    words = []
    with open(vocab_file) as fin:
        for line in fin:
            words.append(line.split()[0])
    return words

def get_data(count, pos_words):
    tie_count = 0
    pos2idx = {}
    with open('pos_vocab_noties.txt', 'wt') as fout:
        with open('pos_y_noties.txt', 'wt') as fout_y:
            for word_idx, word in enumerate(pos_words):
                max_count = 0
                count_list = [(x, y) for x, y in count[word].items()]
                all_count = sum(count[word].values())
                if all_count >= 100:
                    count_list = sorted(count_list, key=lambda x: -x[1])
                    tie = False
                    if len(count_list) > 1:
                        if count_list[0][1] <= all_count/2:
                            tie = True
                        #if count_list[0][1] == count_list[1][1]:
                        #    tie_count += 1
                        #    tie = True
                    if not tie:
                        fout.write(word)
                        fout.write('\n')
                        gold_pos = count_list[0][0]
                        fout_y.write(gold_pos + ' ')
                        if gold_pos not in pos2idx:
                            pos2idx[gold_pos] = len(pos2idx)
                        fout_y.write(str(pos2idx[gold_pos]))
                        fout_y.write('\n')
            

if __name__ == '__main__':
    sent_file = '/data/lily/jk964/models/WSJ/sents/train.txt'
    pos_file = '/data/lily/jk964/models/WSJ/gold_pos/train.txt'
    vocab_file = 'scripts/jabberwocky/pos_vocab.txt'
    words = read_file(sent_file, True)
    pos_tags = read_file(pos_file)
    assert(len(words) == len(pos_tags))
    count = get_count(words, pos_tags)
    for pos in count.keys():
        words = [x[0] for x in sorted([(word, freq) for word, freq in count[pos].items()], key = lambda x: -x[1])]
        print(pos, ' '.join(words[:50]))
