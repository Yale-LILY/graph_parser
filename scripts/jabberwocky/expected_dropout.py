import numpy as np
def read_wcounts(count_file):
    wcounts = {}
    counts = []
    with open(count_file) as fin:
        for line in fin:
            word, count = line.split()
            count = int(count)
            wcounts[word] = count
            counts.append(count)
    counts = np.array(counts)
    return wcounts, counts

def counts2Edp(counts, alpha):
    probs = alpha/(counts+alpha)
    Edp = np.sum((counts/np.sum(counts))*probs)
    return Edp
    

if __name__ == '__main__':
    count_file = '/home/lily/jk964/models/conll2018-task/graph_parser/scripts/jabberwocky/word_counts.txt'
    wcounts, counts = read_wcounts(count_file)
    for alpha in range(2500, 2550):
        Edp = counts2Edp(counts, alpha)
        print(alpha)
        print(Edp)
