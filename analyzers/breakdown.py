from collections import defaultdict
def get_deps(filename):
    gold_deps = []
    dep_sent = []
    with open(filename) as fin:
        for line in fin:
            tokens = line.split()
            if len(tokens) > 5:
                dep = (int(tokens[0]), int(tokens[6]), tokens[7])
                dep_sent.append(dep)
            else:
                gold_deps.append(dep_sent)
    print(len(gold_deps))
    return gold_deps
def compare(gold_deps, pred_deps):
    gold_count = defaultdict(int)
    pred_count = defaultdict(int)
    correct_count = defaultdict(int)
    for gold_sent, pred_sent in zip(gold_deps, pred_deps):
        for gold_dep, pred_dep in zip(gold_sent, pred_sent):
            if gold_dep == pred_dep:
                correct_count[gold_dep[2]] += 1
            gold_count[gold_dep[2]] += 1
            pred_count[pred_dep[2]] += 1
    return gold_count, pred_count, correct_count
def get_f1(gold_count, pred_count, correct_count):
    core = ['nsubj', 'dobj', 'iobj', 'csubj', 'ccomp', 'xcomp']
    gold_others = 0
    pred_others = 0
    correct_others = 0
    for label in gold_count.keys():
        if label in core:
            prec = correct_count[label]/pred_count[label]
            recall = correct_count[label]/gold_count[label]
            print(label)
            print(prec)
            print(recall)
            print(2*(prec*recall)/(prec+recall))
        else:
            gold_others += gold_count[label]
            pred_others += pred_count[label]
            correct_others += correct_count[label]
    print('Others')
    prec = correct_others/pred_others
    recall = correct_others/gold_others
    print(prec)
    print(recall)
    print(2*(prec*recall)/(prec+recall))
        

gold_data = 'data/dev.conllu'
#gold_data = '/data/lily/jk964/models/WSJ_Syntactic_JB/conllu/dev.conllu'
gold_deps = get_deps(gold_data)
jw_data = 'data/jabberwocky_dev.conllu'
#jw_data = 'data/linzen_dev.conllu'
jw_deps = get_deps(jw_data)
gold_count, pred_count, correct_count = compare(gold_deps, jw_deps)
get_f1(gold_count, pred_count, correct_count)
#print(gold_count, pred_count, correct_count)
