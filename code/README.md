# Что под капотом forward ?
Вход:
1. батч предложений ---> (max_length_sentence_de , batch )
2. Батч переводов ---> (max_length_sentence_en , batch )
3. Маска предложений ---> (max_length_sentence_de , max_length_sentence_de)
4. Маска переводов ---> (max_length_sentence_en , max_length_sentence_en)
5. Падинг маска предложений ---> (batch , max_length_sentence_de)
6. Падинг маска переводов ---> (batch , max_length_sentence_en)


1. Транспонированный батч с токенами
2. Транспонированный батч с токенами
3. Изначально подается квадратная матрица из False
4. Верхнетреугольная квадратная матрица: снизу нули,сверху $-\infty$
5. Каждому батчу сопоставляет вектор : сначала False потом True - вероятно это ограничитель по контексту, что нельзя генерировать с самого начала.
6. В том же духе, но для чего - уже не понятно ?

Паддинг маски только для обучения но зачем ?

* FORWARD = ENCODE with padding + DECODE with padding
* DECODER = ENCODE + DECODE
* TRANSLATE = SRC + DECODER


# Что под капотом encode and decode ? И что в декодере ?

# Encode
* ENCODE = TOKEN EMBEDING + POSITIONAL ENCODING + TRANSFORMER ENCODE with SRC MASK
* Принимает предложения и их маски
* Выдает тензор имеющий смысл памяти : (max_len_sentence_de , 1 , emb_size)
# Decode
* DECODE = TOKEN EMBEDING + POSITIONAL ENCODING + TRANSFORMER DECODE with MEMORY and TARGET MASK
* Принимает  ys , memory , tgt_mask , размеров: (27, 1) (27, 1, 128) (27, 27)
* out : (27 ,1 , 128)

# greedy decoder
* В декодере этот out преобразуется в (1 , len_seq , 128)
* (1 , len_seq , 128) ---> [last word emb] --> (1,128) --> generator --> (1,10838) --> max prob ---> (1) - token in vocab

# Что под капотом у evaluate and bleu calculate ?

* EVALUATE = CREATE MASK + FORWARD --> LOGITS
* BLEU CALCULATE = TRANSLATE --> TGT TOKENS + SENTENCE_BLEU

# Что под капотом у TRAIN EPOCH
* TRAIN EPOCH = CREATE MASK + FORWARD --> LOGITS + LOSS FN + BACKWARD

# SUMMARY

* SRC = TOKENS
* TARGET = TOKENS
* SRC MASK and TARGET MASK = NUM TOKENS (SRC or TARGET) x NUM TOKENS (SRC or TARGET)

* SRC and TARGET PADDING MASK = EMBEDING SIZE x NUM TOKENS (SRC or TARGET)

* ENCODE = TOKEN EMBEDING + POSITIONAL ENCODING + TRANSFORMER ENCODE with SRC MASK
* DECODE = TOKEN EMBEDING + POSITIONAL ENCODING + TRANSFORMER DECODE with MEMORY and TARGET MASK
* FORWARD = ENCODE with padding + DECODE with PADDING MASK
* DECODER = ENCODE + DECODE
* TRANSLATE = SRC + DECODER
* EVALUATE = CREATE MASK + FORWARD --> LOGITS
* BLEU CALCULATE = TRANSLATE --> TGT TOKENS + SENTENCE_BLEU
* TRAIN EPOCH = CREATE MASK + FORWARD --> LOGITS + LOSS FN + BACKWARD


# RNN ENCODER

* input : (src_emb , src_mask , src_padding_mask ) : ([27, 128, 256]) ,([27, 27]) ,([128, 27])
* output : ([27, 128, 256])
* where 128 - batch , 256 - emb_size , 27 - src tokens length
* hyperparameters : emb_size, NHEAD, dim_feedforward ,num_encoder_layers
# RNN DECODER
* input (tgt emb ) ([41, 128, 256]) ,

??? masks ([45, 128, 256]) ,([41, 41]) None , ([128, 41]) ,([128, 45])
* output : ([41, 128, 256])
