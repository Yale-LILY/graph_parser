counts = []
with open('word_counts.txt') as fin:
    for line in fin:
        counts.append(int(line.split()[1]))
        if len(counts) == 100:
            print(sum(counts))

