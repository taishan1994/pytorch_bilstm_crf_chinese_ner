import torch
import torch.nn as nn
from torchcrf import CRF

class BiLstmCrf(nn.Module):
    def __init__(self, args, device, embedding_pretrained=None):
        super(BiLstmCrf, self).__init__()
        self.embedding_pretrained = embedding_pretrained
        self.hidden_size = args.hidden_size
        self.device = device
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.build_model(args)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device)
        return h0, c0

    def build_model(self, args):
        if args.use_pretrained:
            self.embedding = nn.Embedding.from_pretrained(self.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(args.embedding_size, args.hidden_size, args.num_layers, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout2)
        self.linear = nn.Linear(args.hidden_size * 2, args.num_tags)
        self.crf = CRF(args.num_tags, batch_first=True)


    def forward(self, inputs, labels, mask):
        inputs = self.embedding(inputs)
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        hidden = self.init_hidden(batch_size)
        out, (hn, _) = self.lstm(inputs, hidden)
        out = out.contiguous().view(-1, self.hidden_size * 2)
        out = self.dropout(out)
        out = self.linear(out)
        out = out.contiguous().view(batch_size, seq_len, -1)
        if labels is None:
            output = self.crf.decode(out , mask=mask)
            return output
        else:
            loss = -self.crf(out, labels, mask=mask, reduction='mean')
            return loss