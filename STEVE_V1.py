""" Word encoder and sentence encoder both work. The TSN layer in the middle does not. """
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

        next_in = Variable(torch.ones(1, hidden.shape[1], embedding_size)).cuda()         # SOS character
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

    def forward(self, hidden, correct_outputs=None):

        next_in = Variable(torch.ones(1, hidden.shape[1], word_decoder_size)).cuda()         # SOS character
        outputs = []

        if correct_outputs is None:
            for i in range(max_sentence_len):
                next_in, hidden = self.rnn(next_in, hidden)
                next_in = self.linear(next_in)
                outputs.append(word_decoder(next_in))
        else:
            for i in range(max_sentence_len):
                out, hidden = self.rnn(next_in, hidden)
                out = self.linear(out)
                outputs.append(word_decoder(out))

                next_in = correct_outputs[i].unsqueeze(0)

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


def train_WRAE(iter_num):
    """ training WordRecurrentAutoEncoder for word embeddings """
    for i in range(int(iter_num)):
        batch = Variable(torch.from_numpy(helper.get_word_batch(100))).long().cuda()

        _, e_hidden = word_encoder(batch)
        out = word_decoder(e_hidden)

        cost = helper.get_word_loss(out, helper.one_hot(batch, helper.vocab_size))
        cost.backward()

        w_e_optim.step()
        w_d_optim.step()

        w_e_optim.zero_grad()
        w_d_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])

        if i % 1000 == 0:
            print(helper.id_vec2seq(batch[:, 0].data), 'to', helper.id_vec2seq([np.argmax(vec) for vec in out[:, 0].data]))

        if i % 2000 == 0:
            torch.save(word_encoder.state_dict(), 'w_e')
            torch.save(word_decoder.state_dict(), 'w_d')


def train_SRAE(iter_num):

    for i in range(int(iter_num)):
        batch, _ = helper.get_sentence_batch(20)

        targets = [word_batch for word_batch in batch]

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

        if i % 2000 == 0:
            torch.save(sentence_encoder.state_dict(), 's_e')
            torch.save(sentence_decoder.state_dict(), 's_d')

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


def train_TSN(tsn_iter_num):
    """ hopefully this is the masterpiece """

    # training tsn network
    for i in range(int(tsn_iter_num)):
        batch, targets = helper.get_sentence_batch(50)

        in_meaning = sentence_encoder(batch)
        ans_meaning = linear_tsn(in_meaning)
        o, _ = sentence_decoder(ans_meaning)

        cost = helper.get_loss(o, targets)
        cost.backward()

        linear_tsn_optim.step()
        linear_tsn_optim.zero_grad()

        if i % 500 == 0:
            print('step', i, 'cost', cost.data[0])

        if i % 2000 == 0:
            torch.save(linear_tsn.state_dict(), 'linear_tsn')

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



max_word_len = 10
max_sentence_len = 120         # in words
helper = Helper.Helper('data.txt', max_sentence_len, max_word_len)
vocab_size = helper.vocab_size
embedding_size = 80

word_encoder_size = 75
word_decoder_size = word_encoder_size * 4

sentence_encoder_size = 700
sentence_decoder_size = sentence_encoder_size * 4

word_encoder = WordEncoder().cuda()
word_decoder = WordDecoder(word_encoder.embeddings).cuda()
sentence_encoder = SentenceEncoder().cuda()
sentence_decoder = SentenceDecoder().cuda()
linear_tsn = LinearTSN().cuda()

w_e_optim = optim.Adam(word_encoder.parameters(), lr=1e-3)
w_d_optim = optim.Adam(word_decoder.parameters(), lr=1e-3)
s_e_optim = optim.Adam(sentence_encoder.parameters(), lr=2e-5)
s_d_optim = optim.Adam(sentence_decoder.parameters(), lr=2e-5)
linear_tsn_optim = optim.Adam(linear_tsn.parameters(), lr=2e-5)

word_encoder.load_state_dict(torch.load('w_e'))
word_decoder.load_state_dict(torch.load('w_d'))

sentence_encoder.load_state_dict(torch.load('s_e'))
sentence_decoder.load_state_dict(torch.load('s_d'))

linear_tsn.load_state_dict(torch.load('linear_tsn'))

train_TSN(2e5)
