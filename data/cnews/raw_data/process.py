import codecs
from collections import Counter

def process(path):
    vocab_tmp = []
    with codecs.open(path,'r',encoding='utf-8') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            line = line.replace('\r','').split(' ')
            if len(line) == 2:
                vocab_tmp.append(line[0])
    return vocab_tmp

vocab = []
paths = ['train.char.bmes','dev.char.bmes','test.char.bmes']
for path in paths:
    vocab_tmp = process(path)
    vocab.extend(vocab_tmp)
word_counter = sorted(Counter(vocab).items(), key=lambda x:x[1], reverse=True)
print('总字数：', len(word_counter))
fp = open('../final_data/vocab.txt','w',encoding='utf-8')
fp.write('[PAD]' + '\n')
fp.write('[UNK]' + '\n')
for k,v in word_counter:
    fp.write(k + '\n')
fp.close()
