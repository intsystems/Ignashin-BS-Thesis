import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.functional import F
import random
import math
import time

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout , bidirectional = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout ,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
        
    def forward(self, src):
        embedded = self.embedding(src)# <YOUR CODE HERE>
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)
        if self.bidirectional :
            output = output[:,:,-output.shape[2]//2:]
            hidden = (hidden[0][1:,:,:], hidden[1][1:,:,:] )
        # print(output.shape , hidden[0].shape , hidden[1].shape)
        return output , hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # print(self.Wa(query).shape , self.Ua(keys).shape)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        
        context = torch.bmm(weights, keys)
        return context, weights
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, without_attention = False):
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.without_attention = without_attention

        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        self.rnn = nn.LSTM(
            input_size=2*hid_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.out = nn.Linear(
            in_features=hid_dim,
            out_features=output_dim
        )
        if not without_attention :
            self.attention = BahdanauAttention(hid_dim)
        self.dropout = nn.Dropout(p=dropout)# <YOUR CODE HERE>
    def forward(self, input, encoder_output , hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))# <YOUR CODE HERE>
        query = hidden[0].permute(1, 0, 2)
        
        encoder_output = encoder_output.permute(1,0,2)
        # print('attention input: query ',query.shape)
        # print('attention input: encoder output ',encoder_output.shape)
        if not self.without_attention :
            context, attn_weights = self.attention(query,encoder_output )
            context = context.permute(1,0,2)
        else :
            context = encoder_output.permute(1,0,2)
        # print('attention output: embedded and context',embedded.shape, context.shape)
        
        input_rnn = torch.cat((embedded, context), dim=2)
        # print('input rnn decoder: input ',input_rnn.shape)
        # print('input rnn decoder: hidden',hidden[0].shape , hidden[1].shape)


        output, hidden = self.rnn( input_rnn , hidden )
        prediction = self.out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder ,device , without_attention=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.without_attention = without_attention
        assert self.decoder.ar
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_output ,hidden = self.encoder(src)
        if self.without_attention :
            encoder_output = torch.sum(encoder_output, dim=0).unsqueeze(0)
            # print(encoder_output.shape)
        input = trg[0,:]
        
        for t in range(1, max_len):
            # print(t)
            # print('decoder input: input, encoder_output, hidden ',input.shape, encoder_output.shape ,hidden[0].shape ,hidden[1].shape)
            output, hidden = self.decoder(input, encoder_output ,hidden)
            # print('decoder output: ' ,output.shape , hidden[0].shape)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
