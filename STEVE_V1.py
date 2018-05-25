""" same as V_0 but WRAE and STEVE in one file  """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random

# import OneHelperToRuleThemAll as Helper
import HelperFor01 as Helper


class WordEncoder(nn.Module):

    def __init__(self):
        super(WordEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, word_encoder_size, bidirectional=True, batch_first=False, num_layers=2)

    def forward(self, x):
        x = self.embeddings(x)
        rnn_out, h = self.rnn(x)      # hidden is the final state
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1).unsqueeze(0)         # [1, batch_size, decoder_size)
        return rnn_out, h


class WordDecoder(nn.Module):

    def __init__(self, embeddings):
        super(WordDecoder, self).__init__()
        self.rnn = nn.GRU(embedding_size, word_decoder_size, bidirectional=False, batch_first=False)
        self.linear = nn.Linear(word_decoder_size, vocab_size)
        self.embeddings = embeddings

    def get_next_input(self, out_logits):
        _, pred = torch.max(out_logits, dim=2)
        next_input = self.embeddings(pred)
        return next_input

    def forward(self, hidden):

        next_in = Variable(torch.ones(1, hidden.shape[1], embedding_size))         # SOS character
        outputs = []

        for i in range(max_word_len):
            out, hidden = self.rnn(next_in, hidden)
            out = self.linear(out)          # we might have to reshape this
            outputs.append(out)

            next_in = self.get_next_input(out)

        return torch.cat(outputs, dim=0)


class SentenceEncoder(nn.Module):

    def __init__(self):
        super(SentenceEncoder, self).__init__()
        self.rnn = nn.GRU(word_decoder_size, sentence_encoder_size, bidirectional=True, batch_first=False, num_layers=2)

    def forward(self, x):

        encoded_words = []
        for i in range(max_sentence_len):
            _, word_vector = word_encoder(x[i])
            encoded_words.append(word_vector)

        encoded_words = torch.cat(encoded_words, dim=0)

        _, h = self.rnn(encoded_words)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1).unsqueeze(0)
        return h


class SentenceDecoder(nn.Module):

    def __init__(self):
        super(SentenceDecoder, self).__init__()
        self.rnn = nn.GRU(word_decoder_size, sentence_decoder_size, bidirectional=False, batch_first=False)
        self.linear = nn.Linear(sentence_decoder_size, word_decoder_size)

    def forward(self, hidden):

        next_in = Variable(torch.ones(1, hidden.shape[1], word_decoder_size))         # SOS character
        outputs = []

        for i in range(max_sentence_len):
            next_in, hidden = self.rnn(next_in, hidden)
            next_in = self.linear(next_in)
            outputs.append(word_decoder(next_in))

        # outputs = torch.cat(outputs, dim=0)
        return outputs, hidden


class LinearTSN(nn.Module):

    def __init__(self):
        super(LinearTSN, self).__init__()
        self.linear0 = nn.Linear(sentence_decoder_size, 2048)
        self.linear1 = nn.Linear(2048, 2560)
        self.linear2 = nn.Linear(2560, sentence_decoder_size)

    def forward(self, x):
        h0 = F.relu(self.linear0(x))
        h1 = F.relu(self.linear1(h0))
        out = F.relu(self.linear2(h1))
        return out


def train0(iter_num):
    """ training everything, including word encoder, together """
    for i in range(int(iter_num)):
        batch, targets = helper.get_sentence_batch(20)

        h = sentence_encoder(batch)
        o, _ = sentence_decoder(h)

        cost = helper.get_loss(o, targets)
        cost.backward()

        w_e_optim.step()
        w_d_optim.step()
        s_e_optim.step()
        s_d_optim.step()

        w_e_optim.zero_grad()
        w_d_optim.zero_grad()
        s_e_optim.zero_grad()
        s_d_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])

            word_list = [word_batch[:, 0].data for word_batch in o]
            example_phrase = ''
            for word in word_list:
                example_phrase += helper.id_vec2seq([np.argmax(vec) for vec in word]) + ' '
            print(example_phrase)

            word_list = [helper.one_hot(torch.squeeze(target, dim=0), helper.vocab_size).data for target in targets]
            example_phrase = ''
            for word in word_list:
                example_phrase += helper.id_vec2seq([np.argmax(vec) for vec in word]) + ' '
            print(example_phrase)


def train1(wrae_iter_num, steve_iter_num):

    # training WRAE
    for i in range(int(wrae_iter_num)):
        batch = helper.get_word_batch(20)

        _, e_hidden = word_encoder(batch)
        out = word_decoder(batch)

        cost = helper.get_word_loss(out, batch)

        w_e_optim.step()
        w_d_optim.step()

        w_e_optim.zero_grad()
        w_d_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])

    # training seq2seq with the sentence vectors
    for i in range(int(steve_iter_num)):
        batch, targets = helper.get_sentence_batch(20)

        h = sentence_encoder(batch)
        o, _ = sentence_decoder(h)

        cost = helper.get_loss(o, targets)
        cost.backward()

        s_e_optim.step()
        s_d_optim.step()

        s_e_optim.zero_grad()
        s_d_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])


def train_WRAE(iter_num):
    """ training WordRecurrentAutoEncoder for word embeddings """
    for i in range(int(iter_num)):
        batch = Variable(torch.from_numpy(helper.get_word_batch(20))).long()

        _, e_hidden = word_encoder(batch)
        out = word_decoder(e_hidden)

        cost = helper.get_word_loss(out, helper.one_hot(batch, helper.vocab_size))

        w_e_optim.step()
        w_d_optim.step()

        w_e_optim.zero_grad()
        w_d_optim.zero_grad()

        if i % 1 == 0:
            print('step', i, 'cost', cost.data[0])
            torch.save(word_encoder.state_dict(), 'w_e')


def train_TSN(wrae_iter_num, srae_iter_num, tsn_iter_num):
    """ hopefully this is the masterpiece """
    # training WRAE
    train_WRAE(wrae_iter_num)

    # training SentenceRecurrentAutoEncoder
    for i in range(int(srae_iter_num)):
        batch, _ = helper.get_sentence_batch(20)

        targets = [word_batch for word_batch in batch]
        print(targets[0].shape)

        h = sentence_encoder(batch)
        o, _ = sentence_decoder(h)

        cost = helper.get_loss(o, batch)
        cost.backward()

        s_e_optim.step()
        s_d_optim.step()

        s_e_optim.zero_grad()
        s_d_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])

    # training tsn network
    for i in range(int(tsn_iter_num)):
        batch, targets = helper.get_sentence_batch(20)

        in_meaning = sentence_encoder(batch)
        ans_meaning = linear_tsn(in_meaning)
        o, _ = sentence_decoder(ans_meaning)

        cost = helper.get_loss(o, targets)
        cost.backward()

        linear_tsn_optim.step()

        linear_tsn_optim.zero_grad()

        if i % 1 == 0:
            print('step', i, 'cost', cost.data[0])


max_word_len = 10
max_sentence_len = 45         # in words
helper = Helper.Helper('data.txt', max_sentence_len, max_word_len)
vocab_size = helper.vocab_size
embedding_size = 80

word_encoder_size = 75
word_decoder_size = word_encoder_size * 4

sentence_encoder_size = 350
sentence_decoder_size = sentence_encoder_size * 4

word_encoder = WordEncoder()
word_decoder = WordDecoder(word_encoder.embeddings)
sentence_encoder = SentenceEncoder()
sentence_decoder = SentenceDecoder()
linear_tsn = LinearTSN()

word_encoder.load_state_dict(torch.load('w_e'))
word_encoder.state
"""word_encoder = torch.load('w_e')
word_decoder = torch.load('w_d')
sentence_encoder = torch.load('s_e')
sentence_decoder = torch.load('s_d')
linear_tsn = LinearTSN()"""

w_e_optim = optim.Adam(word_encoder.parameters(), lr=1e-3)
w_d_optim = optim.Adam(word_decoder.parameters(), lr=1e-3)
s_e_optim = optim.Adam(sentence_encoder.parameters(), lr=5e-4)
s_d_optim = optim.Adam(sentence_decoder.parameters(), lr=5e-4)
linear_tsn_optim = optim.Adam(linear_tsn.parameters(), lr=1e-4)

train_TSN(1, 1, 1)
