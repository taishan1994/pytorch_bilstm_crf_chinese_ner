import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser):
        # args for path
        parser.add_argument('--output_dir', default='../checkpoints/',
                            help='the output dir for model checkpoints')

        parser.add_argument('--pretrained_dir', default='../data/cnews/final_data/wiki_word/',
                            help='pretrained dir for uer')
        parser.add_argument('--pretrained_name', default='wiki.word.embedding.pkl',
                            help='pretrained file')
        parser.add_argument('--data_dir', default='../data/tcner/',
                            help='data dir for uer')
        parser.add_argument('--log_dir', default='../logs/',
                            help='log dir for uer')

        # other args
        parser.add_argument('--num_tags', default=65, type=int,
                            help='number of tags')
        parser.add_argument('--num_layers', default=2, type=int,
                            help='number of kernels')
        parser.add_argument('--seed', type=int, default=123,
                            help='random seed')

        parser.add_argument('--vocab_size', default=352217, type=int,
                            help='vocab_size')
        parser.add_argument('--embedding_size', default=300, type=int,
                            help='embedding_size')
        parser.add_argument('--dropout', default=0.3, type=float,
                            help='dropout')
        parser.add_argument('--dropout2', default=0.5, type=float,
                            help='dropout2')
        parser.add_argument('--hidden_size', default=128, type=int,
                            help='filter_sizes')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--max_seq_len', default=256, type=int)

        parser.add_argument('--eval_batch_size', default=12, type=int)

        # train args
        parser.add_argument('--train_epochs', default=15, type=int,
                            help='Max training epoch')

        # 2e-5
        parser.add_argument('--lr', default=3e-5, type=float,
                            help='learning rate for the bert module')

        # 0.5
        parser.add_argument('--max_grad_norm', default=1, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=32, type=int)
        parser.add_argument('--use_pretrained', default=True, action='store_true',
                            help='whether to use pretrained embedding')

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()