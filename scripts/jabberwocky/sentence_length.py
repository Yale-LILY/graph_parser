import sys, os

base_dir = '/data/lily/jk964/models'
filenames = ['BrownCF', 'BrownCG', 'BrownCK', 'BrownCM', 'BrownCN', 'BrownCR']
#filenames = ['BrownCF']
lens = []
for filename in filenames:
    filename = os.path.join(base_dir, filename, 'sents/test.txt')
    with open(filename) as fin:
        for line in fin:
            sent = line.split()
            lens.append(len(sent))
print(sum(lens))
print(sum(lens)/len(lens))
print(len(lens))
        
