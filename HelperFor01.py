""" Helper for functions in common between STEVE and WRAE, doesn't work with V3 (word2vec similar encodings) """
import torch
import numpy as np
import string
import torch.nn as nn
import random
import torch.nn.functional as F
from torch.autograd import Variable

import CornellMovieCorpusPreprocessor as prep


class Helper:

    def __init__(self, filename, max_sentence_len, max_word_len):
        """
        filename: txt with training data directory, has to change when using decent datasets
        max_sentence_len: maximum number of words in a sentence
        max_word_len: maximum word length in characters
        """

        # setting up vocab
        self.EOS_char = '@'
        self.PAD_char = 'π'
        self.vocab = [self.PAD_char] + [self.EOS_char] + list(string.printable)
        self.vocab_size = len(self.vocab)

        # preparing the dataset for WRAE
        self.data = open(filename).read()
        print('data loaded')
        self.clean_data()
        print('data cleaned')
        self.data = self.data.split()

        self.max_sentence_len = max_sentence_len
        self.max_word_len = max_word_len

        self.EOS_word = ''
        for i in range(max_word_len-1):
            self.EOS_word += self.EOS_char

        self.PAD_word = ''
        for i in range(max_word_len-1):
            self.PAD_word += self.PAD_char

    def clean_data(self):
        """ removes unwanted characters from the data """
        for char in self.data:
            if char not in self.vocab:
                self.data = self.data.replace(char, '')

    def softmax(self, x):
        """Compute softmax of x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def vec2char(self, vec, t=0.5):
        """ returns a character given it's vector (which expressed the probability of each char) and uses temperature to
        increase or diminish randomness. The higher the temperature the more random the output will be (values should range
        between 0 and 1) """
        # l is the ordered list of characters
        vec = np.reshape(vec, [-1]) / t
        vec = self.softmax(vec)
        char = np.random.choice(self.vocab, p=vec)
        return char

    def mat2seq(self, mat, t=0.5):
        """ returns a string of characters given a series of vectors representing their probabilies
        mat.shape = [seq_len, vocab_size] """
        out = ''
        for vec in mat:
            out += self.vec2char(vec, t=t)
        return out

    def seq2id(self, seq):
        """ returns an array of the corresponding ids in the vocab given a string of characters """
        return np.array([self.vocab.index(c) for c in seq])

    def id_vec2seq(self, id_vec):
        """ returns a string of characters given a sequence of ids """
        seq = ''
        for id in id_vec:
            seq += self.vocab[id]
        return seq

    def one_hot(self, batch, depth):
        """ converts a batch of ids (max_word_len, batch_size) to a one_hot batch (max_word_len, batch_size, vocab_size) """
        emb = nn.Embedding(depth, depth)
        emb.weight.data = torch.eye(depth)
        return emb(batch)

    def word_pad(self, seq):
        """ pads a word """
        seq += self.EOS_char
        while len(seq) < self.max_word_len:
            seq += self.PAD_char
        return seq

    def sentence_pad(self, sentence):
        """ pads a sentence with PAD and EOS words """
        word_list = sentence.split()
        word_list.append(self.EOS_word)
        for i in range(self.max_sentence_len - len(word_list)):
            word_list.append(self.PAD_word)
        return word_list

    def is_word_acceptable(self, seq):
        """ checks if a word's length is acceptable and if the characters are in the vocab """
        if len(seq) < self.max_word_len:
            for char in seq:
                if char in self.vocab:      # useless if data has been cleaned
                    pass
                else:
                    print(char)
                    return False
        else:
            return False
        return True

    def is_sentence_acceptable(self, sentence):
        """ checks if the sentence is the right length and if each word is acceptable """
        sentence = sentence.split()
        if len(sentence) < self.max_sentence_len:
            for word in sentence:
                if self.is_word_acceptable(word):
                    pass
                else:
                    return False
        else:
            return False

        return True

    def mse_loss(self, input, target):
        return torch.sum((input - target) ** 2) / input.data.nelement()

    def softmax_cross_entropy_with_logits(self, logits, labels):
        logits = F.log_softmax(logits)
        var = -labels * logits
        return var.sum(1).mean()

    def get_word_loss(self, model_out, labels):
        """ stepwise softmax_cross_entropy with output from WRAE (max_word_len, batch_size, vocab_size) """
        loss = 0
        for i in range(self.max_word_len):
            loss += self.softmax_cross_entropy_with_logits(model_out[i], labels[i])
        return loss / self.max_word_len

    def get_loss(self, outputs, targets):
        """ output and targets are list of word_batches (max_word_len, batch_size, vocab_size) """
        loss = 0
        for w_batch, target in zip(outputs, targets):
            loss += self.get_word_loss(w_batch, self.one_hot(torch.squeeze(target, dim=0), self.vocab_size))
        return loss

    def get_word_batch(self, b_size):
        """ returns a batch of random words from the data file
        the target is the same as the input as it is for the WRAE """
        batch = np.zeros((self.max_word_len, b_size))

        i = 0
        while i < b_size:
            sequence = random.choice(self.data)         # data was split into words

            if self.is_word_acceptable(sequence):
                vector = self.seq2id(self.word_pad(sequence))
                batch[:, i] = vector
                i += 1
        else:
            return batch

    def sentece_batch2ids(self, sentece_batch):
        """ converts a list of padded sentences to a list of word_batches """
        sentences = [[self.word_pad(sentence[i]) for sentence in sentece_batch] for i in range(self.max_sentence_len)]

        for i in range(len(sentences)):
            batch = np.zeros((self.max_word_len, len(sentece_batch)))
            for j in range(len(sentences[i])):
                batch[:, j] = self.seq2id(sentences[i][j])

            sentences[i] = Variable(torch.from_numpy(batch).long().unsqueeze(0))

        return sentences

    def get_sentence_batch(self, b_size):
        """ return a batch of questions and answers (inputs and outputs in general)
        input and target are lists of word batches as ids """
        questions, answers = [], []

        while len(questions) < b_size:
            index = random.randint(0, len(prep.questions)-1)
            question = prep.questions[index]
            answer = prep.answers[index]

            if self.is_sentence_acceptable(question) and self.is_sentence_acceptable(answer):
                questions.append(self.sentence_pad(question))
                answers.append(self.sentence_pad(answer))

        questions = torch.cat(self.sentece_batch2ids(questions), dim=0)
        answers = self.sentece_batch2ids(answers)

        return questions, answers
