import os
import logging
from transformers import BertTokenizer

import config
from utils import utils


logger = logging.getLogger(__name__)


class InputExample:
    def __init__(self, set_type, text, entity, labels=None):
        self.set_type = set_type
        self.text = text
        self.entity=entity
        self.labels = labels


class CharFeature:
    def __init__(self, token_ids, mask_ids, labels=None):
        self.token_ids = token_ids
        self.mask_ids = mask_ids
        self.labels = labels


class Processor:

    @staticmethod
    def read_txt(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.read().strip()
        return raw_examples

    def get_entities(self, text, labels):
        res = []
        for i,label in enumerate(labels):
            if 'B-' in label:
                ent_type = label.split('B-')[1]
                start = i
            elif 'E-' in label and i + 1 < len(labels) and 'O' == labels[i+1]:
                ent = text[start:i+1]
                res.append([ent_type, "".join(ent), start])
            elif 'S-' in label:
                ent_type = label.split('S-')[1]
                ent = text[i]
                res.append([ent_type, "".join(ent), i])
        return res


    def get_examples(self, raw_examples, set_type):
        examples = []
        tmp_char = []
        tmp_label = []
        for line in raw_examples.split('\n'):
            line = line.replace('\r','').split(' ')
            if len(line) == 2:
                tmp_char.append(line[0])
                tmp_label.append(line[1].replace('M-','I-'))
            elif len(line) == 1:
                entity=self.get_entities(tmp_char,tmp_label)
                examples.append(InputExample(set_type=set_type,
                                             text=tmp_char,
                                             entity=entity,
                                             labels=tmp_label))
                tmp_char = []
                tmp_label = []
        return examples


def convert_word_example(ex_idx, example: InputExample, max_seq_len, labels, ent2id, word2id):
    set_type = example.set_type
    raw_text = example.text
    entity = example.entity
    char_labels = example.labels
    # 文本元组
    callback_info = ("".join(raw_text[:max_seq_len]),)
    callback_labels = {x:[] for x in labels}
    for ent in entity:
        callback_labels[ent[0]].append((ent[1],ent[2]))
    callback_info += (callback_labels,)

    label_ids = char_labels
    label_ids = [ent2id[lab] for lab in label_ids]
    if len(label_ids) >= max_seq_len:
        label_ids = label_ids[:max_seq_len]
    else:
        pad_length = max_seq_len - len(label_ids)
        label_ids = label_ids + [0] * pad_length
    assert len(label_ids) == max_seq_len, f'{len(label_ids)}'
    # ========================
    tokens = raw_text
    token_ids = [word2id.get(word, 1) for word in tokens]
    if len(token_ids) >= max_seq_len:
        token_ids = token_ids[:max_seq_len]
        mask_ids = [1] * max_seq_len
    else:
        pad_length = max_seq_len - len(token_ids)
        token_ids = token_ids + [0] * pad_length
        mask_ids = [1] * len(raw_text) + [0] * pad_length
    assert len(token_ids) == max_seq_len, f'{len(token_ids)}'

    if ex_idx < 3:
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f"text: {tokens}")
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"mask_ids: {mask_ids}")
        logger.info(f"labels: {label_ids}")

    feature = CharFeature(
        token_ids=token_ids,
        mask_ids=mask_ids,
        labels=label_ids,
    )

    return feature, callback_info

def convert_examples_to_features(examples, max_seq_len, labels, ent2id, word2id):
    features = []
    callback_info = []

    logger.info(f'Convert {len(examples)} examples to features')

    for i, example in enumerate(examples):
        feature, tmp_callback = convert_word_example(
            ex_idx=i,
            example=example,
            max_seq_len=max_seq_len,
            labels=labels,
            ent2id=ent2id,
            word2id=word2id,
        )
        if feature is None:
            continue

        features.append(feature)
        callback_info.append(tmp_callback)
    logger.info(f'Build {len(features)} features')

    out = (features,)

    if not len(callback_info):
        return out

    out += (callback_info,)
    return out

def get_out(processor, txt_path, args, labels, ent2id, word2id, mode):
    raw_examples = processor.read_txt(txt_path)

    examples = processor.get_examples(raw_examples, mode)
    for i, example in enumerate(examples):
        print(example.text)
        print(example.labels)
        if i == 5:
            break
    out = convert_examples_to_features(examples, args.max_seq_len, labels, ent2id, word2id)
    return out


if __name__ == '__main__':
    args = config.Args().get_parser()
    args.log_dir = './logs/'
    args.max_seq_len = 128
    args.data_dir = './data/cnews/final_data/'
    utils.set_logger(os.path.join(args.log_dir, 'process.log'))
    logger.info(vars(args))

    processor = Processor()

    label2id = {}
    id2label = {}
    with open('./data/cnews/final_data/labels.txt','r') as fp:
        labels = fp.read().strip().split('\n')
    for i,label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label
    print(label2id)

    ent2id = {}
    id2ent = {}
    # 规定从1开始
    with open('./data/cnews/final_data/entities.txt','r') as fp:
        entity_labels = fp.read().strip().split('\n')
    for i,label in enumerate(entity_labels):
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
    # print(word2id)

    train_out = get_out(processor, './data/cnews/raw_data/train.char.bmes', args, labels, ent2id, word2id, 'train')
    train_features, tran_callback_info = train_out
    print(tran_callback_info)
    dev_out = get_out(processor, './data/cnews/raw_data/dev.char.bmes', args, labels, ent2id, word2id,  'dev')
    test_out = get_out(processor, './data/cnews/raw_data/test.char.bmes', args, labels, ent2id, word2id, 'test')