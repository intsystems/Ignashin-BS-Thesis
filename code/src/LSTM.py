
import torch 
import torch.nn as nn
from torch import Tensor

from src.transformer import TokenEmbedding ,PositionalEncoding

class RNNencoder(nn.Module):
    def __init__(self, emb_dim=None, hidden_dim=None ):
        super(RNNencoder, self).__init__()
        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim , batch_first=True )

    def forward(self, src: Tensor, src_padding_mask = None):

        # print(src.shape ,src_padding_mask.shape)
        if src_padding_mask is not None :
            length = (~src_padding_mask).sum(1).int().numpy()
            act = src.permute(1,0,2)
            
            x = nn.utils.rnn.pack_padded_sequence(act, length, batch_first=True, enforce_sorted=False)
            out, hidden = self.lstm(x)
            out, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else :
            act = src.permute(1,0,2)
            out, hidden = self.lstm(act)

        return out, hidden
    

class RNNdecoder(torch.nn.Module):
    def __init__(self, emb_dim=None, hidden_dim=None):
        super(RNNdecoder, self).__init__()

        # self.attention = torch.nn.MultiheadAttention(emb_dim, 1)

        self.lstm = torch.nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, input, memory, hidden ,tgt_padding_mask = None): # memory для attention
        
        if tgt_padding_mask is not None :
            length = (~tgt_padding_mask).sum(1).int().numpy()
            act = input.permute(1,0,2)
            x = nn.utils.rnn.pack_padded_sequence(act, length, batch_first=True, enforce_sorted=False)
            out, hidden = self.lstm(x , hidden)
            out, lengths = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else :
            act = input.permute(1,0,2)
            out, hidden = self.lstm(act , hidden)
        # act, _ = self.attention(act.transpose(0, 1),
        #                         encoder_outputs.transpose(0, 1),
        #                         encoder_outputs.transpose(0, 1))
        # result = []
        # for i in range(input.shape[0]) :
        #     act = input[i:i+1,:,:]
        #     act = act.permute(1,0,2)
        #     act, hidden = self.lstm(act,  (hidden[0] , hidden[1]))
        #     result.append(act)

        # result = torch.stack(result).squeeze(2)

        return out
    
class Seq2SeqRNN(nn.Module):
    def __init__(self,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 hidden_size:int = 512, dropout:float = 0.1):
        super(Seq2SeqRNN, self).__init__()

        self.rnn_encoder = RNNencoder(emb_size,hidden_size) # DO IT
        self.rnn_decoder = RNNdecoder(emb_size, hidden_size)# DO IT


        self.generator = nn.Linear(hidden_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor): # src_mask ,tgt_mask ,memory_key_mask - не используются
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory, hidden  = self.rnn_encoder(src_emb , src_padding_mask)
        outs = self.rnn_decoder(tgt_emb, memory, hidden , tgt_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_padding_mask = None): # src_mask - не используется
        return self.rnn_encoder(self.positional_encoding(self.src_tok_emb(src)) , src_padding_mask)

    def decode(self, tgt: Tensor, memory: Tensor,   hidden : Tensor , tgt_padding_mask = None ):
        # print(tgt_mask)
        return self.rnn_decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory , hidden , tgt_padding_mask)
