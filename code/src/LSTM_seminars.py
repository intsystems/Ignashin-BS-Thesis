

import torch 
import torch.nn as nn
from torch import Tensor

        
class Encoder(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, vocab_size, emb_dim=30, hidden_dim=30):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, input):
        r'''
        :param input: тензор размера batch_size x seq_len --- список токенов

        '''
        act = self.embedding(input)
        act, hidden = self.lstm(act)
        return act, hidden


class Decoder(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, vocab_size, emb_dim=30, hidden_dim=30):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim)
        self.attention = torch.nn.MultiheadAttention(emb_dim, 1)

        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True)

        self.linear = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, encoder_outputs, hidden):
        r'''
        :param input: тезор размера batch_size x seq_len
        '''
        act = self.embedding(input)

        act, _ = self.attention(act.transpose(0, 1), 
                                encoder_outputs.transpose(0, 1), 
                                encoder_outputs.transpose(0, 1))

        act = act.transpose(0, 1)
        act, hidden = self.lstm(act, hidden)
        return self.linear(act), hidden

class seq2seq(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(self, vocab_size, emb_dim=30, hidden_dim=30):
        super(seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, emb_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_dim)

    def forward(self, input, decoder_input=None, max_seq_len=64):
        r'''
        '''
        encoder_output, hidden = self.encoder(input)

        if decoder_input is None:
            translated_scores = torch.zeros(len(input), 
                                            max_seq_len, 
                                            self.vocab_size).to(self.device)
            translated_scores[:, 0, input[:, 0]] = 1.
            for i in range(1, max_seq_len):
                translated_scores[:, i:i+1], hidden = self.decoder(
                    torch.argmax(translated_scores[:, 0:i], axis=-1), 
                    encoder_output, 
                    hidden)
        else:
            translated_scores, _ = self.decoder(
                decoder_input, encoder_output, hidden)

        return translated_scores