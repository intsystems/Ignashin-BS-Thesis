import io
import torch
from collections import Counter
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.utils import download_from_url, extract_archive

from src.transformer import TokenEmbedding, PositionalEncoding
from src.train import create_mask


def download_data():
    url_base = (
        "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    )
    train_urls = ("train.de.gz", "train.en.gz")
    val_urls = ("val.de.gz", "val.en.gz")
    test_urls = ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz")

    train_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in train_urls
    ]
    val_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in val_urls
    ]
    test_filepaths = [
        extract_archive(download_from_url(url_base + url))[0] for url in test_urls
    ]
    return train_filepaths, val_filepaths, test_filepaths


def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])


def build_train_vocab(train_filepaths):
    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
    en_vocab = build_vocab(train_filepaths[1], en_tokenizer)
    de_vocab.set_default_index(de_vocab["<unk>"])
    en_vocab.set_default_index(en_vocab["<unk>"])
    return de_vocab, en_vocab, de_tokenizer, en_tokenizer


def data_process(filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for raw_de, raw_en in zip(raw_de_iter, raw_en_iter):
        # print()
        de_tensor_ = torch.tensor(
            [de_vocab[token] for token in de_tokenizer(raw_de.rstrip("\n"))],
            dtype=torch.long,
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en.rstrip("\n"))],
            dtype=torch.long,
        )
        data.append((de_tensor_, en_tensor_))
    return data


def get_train_test_val(
    train_filepaths,
    test_filepaths,
    val_filepaths,
    de_vocab,
    en_vocab,
    de_tokenizer,
    en_tokenizer,
):
    train_data = data_process(
        train_filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer
    )
    val_data = data_process(
        val_filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer
    )
    test_data = data_process(
        test_filepaths, de_vocab, en_vocab, de_tokenizer, en_tokenizer
    )
    return train_data, test_data, val_data


def check_tokens(data, index, de_vocab, en_vocab):
    return {
        "tokens_de": data[index][0],
        "tokens_en": data[index][1],
        "sentence_de": list([de_vocab.get_itos()[j] for j in data[index][0]]),
        "sentence_en": list([en_vocab.get_itos()[j] for j in data[index][1]]),
    }


def tokens_to_sentence(tokens, vocab):
    return [vocab.get_itos()[j] for j in tokens]


def visualize_iter_data(data_iter):
    en_batch = None
    de_batch = None
    for j in data_iter:
        de_batch = j[0]
        en_batch = j[1]
        print("en shape , de shape : ", en_batch.shape, de_batch.shape)
        for i in range(3):
            print(
                tokens_to_sentence(de_batch.permute(1, 0)[i], de_vocab),
                tokens_to_sentence(en_batch.permute(1, 0)[i], en_vocab),
            )

        return en_batch, de_batch


def generate_batch(data_batch, BOS_IDX, PAD_IDX, EOS_IDX):
    de_batch, en_batch = [], []
    for de_item, en_item in data_batch:
        de_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0
            )
        )
        en_batch.append(
            torch.cat(
                [torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0
            )
        )
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return de_batch, en_batch


def get_embed(**kwargs):
    train_iter = kwargs["train_iter"]
    EMB_SIZE = kwargs["EMB_SIZE"]
    DEVICE = kwargs["DEVICE"]
    SRC_VOCAB_SIZE = kwargs["SRC_VOCAB_SIZE"]
    PAD_IDX = kwargs["PAD_IDX"]

    for idx, (src, tgt) in enumerate(train_iter):
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, pad_idx=PAD_IDX, device=DEVICE
        )
        src_emb = PositionalEncoding(EMB_SIZE, dropout=0.1)(
            TokenEmbedding(SRC_VOCAB_SIZE, EMB_SIZE)(src)
        )
        tgt_emb = PositionalEncoding(EMB_SIZE, dropout=0.1)(
            TokenEmbedding(SRC_VOCAB_SIZE, EMB_SIZE)(tgt_input)
        )
        return {
            "src": src,
            "tgt": tgt,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_padding_mask": src_padding_mask,
            "tgt_padding_mask": tgt_padding_mask,
            "src_emb": src_emb,
            "tgt_emb": tgt_emb,
        }
