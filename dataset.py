import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset, DataLoader
# 这里要显示的引入BertFeature，不然会报错
from process import CharFeature
from process import get_out, Processor
import config


class NerDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.mask_ids = [torch.tensor(example.mask_ids, dtype=torch.uint8) for example in features]
        self.labels = [torch.tensor(example.labels) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):

        data = {
            'token_ids':self.token_ids[index],
            'mask_ids':self.mask_ids[index],
            'labels':self.labels[index],
        }

        return data

if __name__ == '__main__':
    args = config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 128

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('./data/cnews/final_data/labels.txt', 'r') as fp:
        labels = fp.read().strip().split('\n')
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    print(label2id)

    ent2id = {}
    id2ent = {}
    with open('./data/cnews/final_data/entities.txt', 'r') as fp:
        entity_labels = fp.read().strip().split('\n')
    for i, label in enumerate(entity_labels):
        ent2id[label] = i+1
        id2ent[i+1] = label
    print(ent2id)

    word2id = {}
    id2word = {}
    with open('./data/cnews/final_data/vocab.txt', 'r') as fp:
        words = fp.read().strip().split('\n')
    for i, word in enumerate(words):
        word2id[word] = i
        id2word[i] = word

    train_out = get_out(processor, './data/cnews/raw_data/train.char.bmes', args, labels, ent2id, word2id, 'train')
    train_features, tran_callback_info = train_out
    train_dataset = NerDataset(train_features)
    for data in train_dataset:
        print(data['token_ids'])
        print(data['mask_ids'])
        print(data['labels'])
        break