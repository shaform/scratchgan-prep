import sys


best_checkpoint = None
best_ppl = None
with open(sys.argv[1]) as infile:
    for line in infile:
        tks = line.split(',')
        ppl = float(tks[1])
        if best_ppl is None or ppl < best_ppl:
            best_ppl = ppl
            best_checkpoint = line


print(best_checkpoint)
print(line)
