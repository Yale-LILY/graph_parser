from tarjan import Tarjan
import numpy as np
from converters import read_sents
if __name__ == '__main__':
    input_file = 'sample_data/predicted_arcs/dev.txt'
    #input_file = 'sample_data/predicted_arcs_greedy/dev.txt'
    #input_file = '/data/lily/jk964/active_projects/tag/pete/predicted_arcs/test_texts.txt'
    #input_file = '/data/lily/jk964/active_projects/ud/graph_parser/data/tag_wsj/predicted_arcs/dev.txt'
    predicted_arcs = read_sents(input_file)
    for sent_idx in range(len(predicted_arcs)):
        predictions = np.array([0] + list(map(int, predicted_arcs[sent_idx])))
        test = Tarjan(predictions, np.arange(1, len(predictions)))
        for cycle in test._SCCs:
            if len(cycle) > 1:
                #print(sent_idx, cycle)
                #print(predictions)
                if len(predictions) <= 9:
                    print(sent_idx, len(predictions))
        #print(test._SCCs)
        #print(test._edges)
        #print(test._vertices)
