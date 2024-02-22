import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Делаем, так чтобы в обучении не было заглядывания на дальнешие слова
def generate_square_subsequent_mask(sz, **kwargs):
    device = kwargs["device"]
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, **kwargs):
    pad_idx = kwargs["pad_idx"]
    device = kwargs["device"]

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(sz=tgt_seq_len, device=device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol, num_samples=1, **kwargs):
    DEVICE = kwargs["DEVICE"]

    src = src.to(DEVICE)
    src = torch.cat([src] * num_samples, dim=1)
    src_mask = src_mask.to(DEVICE)

    if kwargs['transformer'] :  
        memory = model.encode(src, src_mask)
    else :
        # print((~src_mask).sum(1).int().numpy())
        # print(src_mask.shape)
        # print(src.shape)
        # print('Rnn encoding...')
        memory, hidden = model.encode(src= src)
        # print('Rnn encoded.')    
    # print(memory.shape)
    
    ys = torch.ones(1, num_samples).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        # print(f'{i} iteration in bleu decoding')
        memory = memory.to(DEVICE)
        memory_mask = (
            torch.zeros(ys.shape[0], memory.shape[0]).to(DEVICE).type(torch.bool)
        )
        tgt_mask = (generate_square_subsequent_mask(ys.size(0) , device =DEVICE ).type(torch.bool)).to(
            DEVICE
        )
        # print(ys.shape , memory.shape , tgt_mask.shape)

        if kwargs['transformer'] :
            out = model.decode(ys, memory, tgt_mask)
        else :
            # print('Rnn decoding...')
            out = model.decode(ys, memory ,hidden)[:,-1:]
            # print('Rnn decoded.')
        # print(out.shape)
        # print(out)
        # print('out shape ', out.shape)
        out = out.transpose(0, 1)
        # print(f'out shape {out.shape}')
        prob = model.generator(out[:, -1])
        # print(f'prob shape {prob.shape}')
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.detach()
        # print(f'{next_word.shape} next word shape, next_word shape: {next_word.shape}')
        ys = torch.cat([ys, next_word.view(1, -1)], dim=0)
        # print(f' end iteration ys shape: {ys.shape}')
    return ys.transpose(0, 1)


def sampling_decode(
    model, src, src_mask, max_len, start_symbol, num_samples=1, **kwargs
):

    DEVICE = kwargs["DEVICE"]

    src = src.to(DEVICE)
    src = torch.cat([src] * num_samples, dim=1)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, num_samples).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        memory_mask = (
            torch.zeros(ys.shape[0], memory.shape[0]).to(DEVICE).type(torch.bool)
        )
        tgt_mask = (generate_square_subsequent_mask(ys.size(0) , device = DEVICE).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        next_word = torch.multinomial(torch.nn.functional.softmax(prob, dim=-1), 1)
        next_word = next_word.detach()

        ys = torch.cat([ys, next_word.view(1, -1)], dim=0)
    return ys.transpose(0, 1)


def translate(
    model,
    srcs,
    src_vocab,
    tgt_vocab,
    src_tokenizer,
    decoder=greedy_decode,
    ret_tokens=False,
    ret_idx=False,
    max_len_add=10,
    input_idx=False,
    **kwargs
):
    model.eval()
    itos = tgt_vocab.get_itos()
    global_answers = []

    EOS_IDX = kwargs["EOS_IDX"]
    # PAD_IDX = kwargs['PAD_IDX']
    DEVICE = kwargs['DEVICE']
    BOS_IDX = kwargs["BOS_IDX"]

    for src in srcs:
        if not input_idx:
            tokens = (
                [BOS_IDX]
                + [src_vocab.stoi[tok] for tok in src_tokenizer(src)]
                + [EOS_IDX]
            )
            src = torch.LongTensor(tokens)
        num_tokens = len(src)
        src = src.reshape(num_tokens, 1)

        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = decoder(
            model,
            src,
            src_mask,
            max_len=num_tokens + max_len_add,
            start_symbol=BOS_IDX,
            DEVICE = DEVICE ,
            transformer = kwargs['transformer']
        )

        answers = []
        for tgt_token in tgt_tokens:
            if not ret_idx:
                reference = []
                for tok in tgt_token:
                    if tok.item() == tgt_vocab["<eos>"]:
                        break
                    if tok.item() not in {
                        tgt_vocab["<eos>"],
                        tgt_vocab["<bos>"],
                        tgt_vocab["<pad>"],
                    }:
                        reference.append(itos[tok])
                answers.append(" ".join(reference).strip())
                if ret_tokens:
                    answers[-1] = answers[-1].split(" ")
            else:
                reference = []
                for tok in tgt_token:
                    if tok.item() == tgt_vocab["<eos>"]:
                        break
                    if tok.item() not in {
                        tgt_vocab["<eos>"],
                        tgt_vocab["<bos>"],
                        tgt_vocab["<pad>"],
                    }:
                        reference.append(tok.item())

                answers.append(reference)
        global_answers.append(answers)
    return global_answers


def evaluate(model, val_iter, **kwargs):
    model.eval()
    losses = 0

    loss_fn = kwargs["loss_fn"]
    DEVICE = kwargs["DEVICE"]
    pad_idx = kwargs['pad_idx']

    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input , pad_idx = pad_idx , device = DEVICE 
        )
        # print('before model')
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        # print('after model')
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

        # ------------------------
        if idx > 5 :
            break

    return losses / len(val_iter)


def bleu_calculate(model, data_iter, decoder=greedy_decode, **kwargs):
    model.eval()
    bleus = []

    en_vocab = kwargs["en_vocab"]
    de_vocab = kwargs["de_vocab"]
    de_tokenizer = kwargs["de_tokenizer"]
    EOS_IDX = kwargs["EOS_IDX"]
    # PAD_IDX = kwargs['PAD_IDX']
    BOS_IDX = kwargs["BOS_IDX"]
    DEVICE = kwargs["DEVICE"]

    itos = en_vocab.get_itos()
    for idx, (src, tgt) in enumerate(data_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src_input = src.transpose(0, 1)
        tgt_input = tgt.transpose(0, 1)
        tgt_output = translate(
            model,
            src_input,
            de_vocab,
            en_vocab,
            de_tokenizer,
            decoder=decoder,
            ret_tokens=True,
            ret_idx=False,
            input_idx=True,
            num_samples=1,
            EOS_IDX = EOS_IDX,
            BOS_IDX = BOS_IDX ,
            DEVICE =DEVICE ,
            transformer = kwargs['transformer']
        )
        for refs, candidates in zip(tgt_input, tgt_output):
            reference = []
            for tok in refs[1:]:
                if tok.item() == en_vocab["<eos>"]:
                    break
                if tok.item() not in {
                    en_vocab["<eos>"],
                    en_vocab["<bos>"],
                    en_vocab["<pad>"],
                }:

                    reference.append(itos[tok])
            bleus.append(
                sentence_bleu(
                    [reference],
                    candidates[0],
                    smoothing_function=SmoothingFunction().method1,
                )
            )
        # --------------------------------------ATTENTION
        if idx > 5 :
            break
    return np.mean(bleus)


def train_epoch(model, train_iter, optimizer, **kwargs):
    model.train()
    losses = 0

    DEVICE = kwargs["DEVICE"]
    loss_fn = kwargs["loss_fn"]
    pad_idx = kwargs['pad_idx']
    # print('1')
    for idx, (src, tgt) in enumerate(train_iter):
        # print('1')
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # print('1')
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input , pad_idx = pad_idx , device = DEVICE 
        )
        # print('1')
        # print(src.shape,tgt_input.shape,
        #                src_mask.shape,
        #                tgt_mask.shape,
        #                src_padding_mask.shape,
        #                tgt_padding_mask.shape,
        #                src_padding_mask.shape )
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        # print('1')
        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        # print('1')
        optimizer.step()
        losses += loss.item()

        # ---------------------ATTENTION
        if idx > 5 :
            break

        # break # ---------------------------------------------------------------------------------ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!----------------------------------
    return losses / len(train_iter)
