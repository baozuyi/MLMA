import sys

if len(sys.argv) == 1:
    print('python build_vocab.py vocab_num < train_file > vocab_file')
    sys.exit()

d = {}
for s in sys.stdin:
    for w in s.split():
        d[w] = d.get(w,0) + 1

n = int(sys.argv[1])
print('<eos>')
print('<unk>')
i = 2
for k,v in sorted(d.items(),key=lambda d:d[1],reverse=True):
    if k == '<eos>' or k == '<unk>':
        continue
    print(k)
    i += 1
    if i >= n:
        break
