import os
import logging
import numpy as np
import torch
import pickle
from utils import utils, decodeUtils, metricsUtils
import config
import dataset
# 显式传入
from process import CharFeature, get_out, Processor
from models import BiLstmCrf
from dataset import NerDataset
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

args = config.Args().get_parser()
utils.set_seed(args.seed)
logger = logging.getLogger(__name__)
utils.set_logger(os.path.join(args.log_dir, 'bilstm_crf.log'))


class BiLstmCrfForNer:
    def __init__(self,
                 args,
                 train_loader,
                 dev_loader,
                 test_loader,
                 labels=None,
                 id2ent=None,
                 dev_callback_info=None,
                 test_callback_info=None,
                 ):
        super(BiLstmCrfForNer, self).__init__()
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        if args.use_pretrained:
            pretrained_embedding = torch.from_numpy(
                pickle.load(open(os.path.join(args.pretrained_dir, args.pretrained_name), 'rb'))).to(torch.float32)
            self.model = BiLstmCrf(args, self.device, pretrained_embedding)
        else:
            self.model = BiLstmCrf(args, self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.id2ent = id2ent
        self.labels = labels
        self.dev_callback_info = dev_callback_info
        self.test_callback_info = test_callback_info
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    def train(self):
        t_total = len(train_loader) * self.args.train_epochs
        global_step = 0

        eval_steps = 100 #每多少个step打印损失及进行验证
        best_dev_micro_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(self.device)
                loss = self.model(batch_data['token_ids'], batch_data['labels'], batch_data['mask_ids'])
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                global_step += 1
                logger.info("【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, t_total, loss.item()))
                if global_step % eval_steps == 0:
                    dev_loss, precision, recall, micro_f1 = self.dev(
                        self.labels,
                        self.dev_callback_info,
                        self.id2ent,
                    )
                    logger.info("【dev】 loss：{:.6f} precision：{:.4f} recall：{:.4f} micro_f1：{:.4f}".format(dev_loss, precision, recall, micro_f1))
                    if micro_f1 > best_dev_micro_f1:
                        logger.info("------------>保存当前最好的模型")
                        checkpoint = {
                            'epoch': epoch,
                            'loss': dev_loss,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        best_dev_micro_f1 = micro_f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        self.save_ckp(checkpoint, checkpoint_path)

    def dev(self, labels, dev_callback_info, id2ent):
        self.model.eval()
        with torch.no_grad():
            tot_dev_loss = 0.0
            batch_output_all = []
            for eval_step, dev_batch_data in enumerate(self.dev_loader):
                for key in dev_batch_data.keys():
                    dev_batch_data[key] = dev_batch_data[key].to(self.device)
                dev_loss = self.model(dev_batch_data['token_ids'],
                                             dev_batch_data['labels'],
                                             dev_batch_data['mask_ids'])

                # tot_dev_loss += torch.sum(dev_loss).item()
                tot_dev_loss += dev_loss.item()

                batch_output = self.model(dev_batch_data['token_ids'],
                                             None,
                                             dev_batch_data['mask_ids'])
                if len(batch_output_all) == 0:
                    batch_output_all = batch_output
                else:
                    batch_output_all = np.append(batch_output_all, batch_output, axis=0)

            total_count = [0 for _ in range(len(labels))]
            role_metric = np.zeros([len(labels), 3])

            for pred_label, tmp_callback in zip(batch_output_all, dev_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(labels), 3])
                pred_entities = decodeUtils.bioes_decode(pred_label, text, id2ent)
                for idx, _type in enumerate(labels):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric

            mirco_metrics = np.sum(role_metric, axis=0)
            mirco_metrics = metricsUtils.get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
            # print('[eval] precision={:.4f} recall={:.4f} f1_score={:.4f}'.format(mirco_metrics[0], mirco_metrics[1],mirco_metrics[2]))
        return tot_dev_loss, mirco_metrics[0], mirco_metrics[1],mirco_metrics[2]

    def test(self, model_path, labels, test_callback_info, id2ent):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, model_path)
        model.eval()
        pred_label = []
        tot_test_loss = 0.0
        with torch.no_grad():
            for eval_step, test_batch_data in enumerate(self.test_loader):
                for key in test_batch_data.keys():
                    test_batch_data[key] = test_batch_data[key].to(self.device)
                test_loss = model(test_batch_data['token_ids'], test_batch_data['labels'],
                                  test_batch_data['mask_ids'])
                tot_test_loss += test_loss.item()

                batch_output = model(test_batch_data['token_ids'],
                                          None,
                                          test_batch_data['mask_ids'])
                if len(pred_label) == 0:
                    pred_label = batch_output
                else:
                    pred_label = np.append(pred_label, batch_output, axis=0)

            total_count = [0 for _ in range(len(labels))]
            role_metric = np.zeros([len(labels), 3])
            for pred, tmp_callback in zip(pred_label, test_callback_info):
                text, gt_entities = tmp_callback
                tmp_metric = np.zeros([len(labels), 3])
                assert len(pred) == len(text)
                pred_entities = decodeUtils.bioes_decode(pred, text, id2ent)
                for idx, _type in enumerate(labels):
                    if _type not in pred_entities:
                        pred_entities[_type] = []
                    total_count[idx] += len(gt_entities[_type])
                    tmp_metric[idx] += metricsUtils.calculate_metric(gt_entities[_type], pred_entities[_type])

                role_metric += tmp_metric
            id2label = {}
            for i,label in enumerate(labels):
                id2label[i] = label
            logger.info(metricsUtils.classification_report(role_metric, labels, id2label, total_count))

    def predict(self, model_path, text, word2id, max_seq_len, id2ent):
        model = self.model
        optimizer = self.optimizer
        model, optimizer, epoch, loss = self.load_ckp(model, optimizer, model_path)
        for name,param in model.named_parameters():
            print(name)
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            token_ids = [word2id.get(word, 1) for word in text]
            token_ids = token_ids + [0] * (max_seq_len - len(text))
            token_ids = torch.from_numpy(np.array(token_ids)).unsqueeze(0).to(self.device)
            mask_ids = [1] * len(text) + [0] * (max_seq_len - len(text))
            mask_ids = torch.from_numpy(np.array(mask_ids).astype(np.uint8)).unsqueeze(0).to(self.device)
            output = model(token_ids,
                             None,
                             mask_ids)
            pred_entities = decodeUtils.bioes_decode(output[0], text[:max_seq_len], id2ent)
            print(text)
            print(pred_entities)

if __name__ == '__main__':
    args.use_pretrained = False
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


    train_out = get_out(processor, './data/cnews/raw_data/train.char.bmes', args, labels, ent2id, word2id, 'train')
    train_features, train_callback_info = train_out
    train_dataset = NerDataset(train_features)
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              sampler=train_sampler,
                              num_workers=2)

    dev_out = get_out(processor, './data/cnews/raw_data/dev.char.bmes', args, labels, ent2id, word2id, 'dev')
    dev_features, dev_callback_info = dev_out
    dev_dataset = NerDataset(dev_features)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    test_out = get_out(processor, './data/cnews/raw_data/test.char.bmes', args, labels, ent2id, word2id, 'test')
    test_features, test_callback_info = test_out
    test_dataset = NerDataset(test_features)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.eval_batch_size,
                            num_workers=2)

    biLstmCrfForNer = BiLstmCrfForNer(
        args,
        train_loader,
        dev_loader,
        test_loader,
        labels,
        id2ent,
        dev_callback_info,
        test_callback_info,
    )
    biLstmCrfForNer.train()
    model_path = './checkpoints/best.pt'
    biLstmCrfForNer.test(model_path, labels, test_callback_info, id2ent)
    raw_text = "虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。"
    # 真实标签："labels": [["T0", "NAME", 0, 3, "虞兔良"], ["T1", "RACE", 17, 19, "汉族"], ["T2", "CONT", 20, 24, "中国国籍"], ["T3", "LOC", 34, 39, "浙江绍兴人"], ["T4", "TITLE", 40, 44, "中共党员"], ["T5", "EDU", 45, 48, "MBA"], ["T6", "TITLE", 49, 52, "经济师"]]
    # 预测标签：{'NAME': [('虞兔良', 0)], 'RACE': [('汉族', 17)], 'CONT': [('中国国籍', 20)], 'TITLE': [('中共党员', 40), ('经济师', 49)], 'EDU': [('MBA', 45)]}
    biLstmCrfForNer.predict(model_path, raw_text, word2id, args.max_seq_len, id2ent)
