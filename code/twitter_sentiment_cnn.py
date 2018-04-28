#!/usr/bin/env python

import numpy as np
import re, string 
from typing import Tuple, List, Dict
from vocabulary import *
from itertools import chain
from keras.models import Sequential 
import keras.optimizers
from keras.layers import Dense, Merge, Embedding, Conv1D, MaxPooling1D, Flatten, LSTM
from keras.callbacks import EarlyStopping

def read_data(filename):
    labels = []
    sentences = []
    sentences_str = []
    tags = []
    IDs = []

    max_tags_num = 0
    max_sentence_len = 0


    with open(filename) as f:
        f.readline()
        for line in f:

            vals = line.strip().split('\t')
            ID = vals[0].strip()
            sentence_str = vals[1].strip()
            sentence_str = sentence_str.replace(':(', "sad")
            sentence_str = sentence_str.replace('T_T', "sad")
            sentence_str = sentence_str.replace(':)', "joy")
            sentence_str = sentence_str.replace(':D', "joy")
            sentence_str = sentence_str.replace('^^', "joy")
            sentence_str = sentence_str.replace(';-)', "cute")
            sentence_str = sentence_str.replace(':P', "blink")

            # split at each punctuation, remove punctuation at the meanwhile
            sentences_words = re.split(r'\\n| |%|&|\'|\"|,|:|;|!|_|=|\.|\(|\)|\$|\?|\*|\+|\-|\]|\[|\{|\}|\\|\/|\||\<|\>|\^|\`|\~', sentence_str)  

            tag = [] 
            istag = 1
            sentences_words.reverse()    # when words with # locates at the end of sentence, they are considered as tags

            for i, word in enumerate(sentences_words):  
                if len(word) == 0:
                    sentences_words.remove(word)
                    i -= 1

                elif word[0] is '#' and istag == 1:    #tags
                    tag.append(word[1:])
                    sentences_words.remove(word)
                    i -= 1
                    
                elif word[0] is '#' and istag == 0:   #words in sentence
                    sentences_words[i] = word[1:]

                elif word[0] is '@':
                    sentences_words[i] = "@NAME"      #someone's name
                    istag = 0

                elif any(char.isdigit() for char in word):
                    sentences_words[i] = "xnumx"
                    istag = 0

                else:
                    istag = 0 
                    
            sentences_words.reverse()
            sentences.append(sentences_words)
            sentences_str.append(sentence_str)
            tags.append(tag)
            labels.append(vals[2:13])
            IDs.append(ID)
            
       
            if len(tag) > max_tags_num:
                max_tags_num = len(tag) 

            if len(sentences_words) > max_sentence_len:
                max_sentence_len = len(sentences_words)
     
        labels = np.array(labels)

  
    return IDs, sentences_str, sentences, tags, labels, max_tags_num, max_sentence_len


class load_data():
    PAD = "@PADDING"
    OOV = "</s>"
    NAME = "@NAME"
    OOV_ID = 0
    NAME_ID = -1

    def __init__(self, type='train'):

        dataset_file = "2018-E-c-En-" + type + ".txt"
        w2vfile = "vectors.goldbergdeps.txt"

        if type is 'train':
            _, _, self.sentences_words, self.tags_words, self.labels, self.max_tags_num, self.max_sentence_len = read_data(dataset_file)

            self.word_vocab = Vocabulary()
            for word in chain.from_iterable(zip(*self.sentences_words)):
                self.word_vocab.add(word)
            for word in chain.from_iterable(zip(*self.tags_words)):
                self.word_vocab.add(word)
            
            self.word_vocab.add("@PADDING", 0)

            # print('word_vocab——size')
            # print(self.word_vocab.size())
            vocab_file = "vocabulary_train.txt"
            self.word_vocab.to_file(vocab_file)

            maxlen_file = "maxlen_train.txt"
            with io.open(maxlen_file, 'w', encoding='utf8') as f:
                f.write(str(self.max_sentence_len) + '\t' + str(self.max_tags_num))

        else:

            vocab_file = "vocabulary_train.txt"
            self.word_vocab = Vocabulary.from_file(vocab_file)

            maxlen_file = "maxlen_train.txt"
            with io.open(maxlen_file, encoding='utf8') as f:
                for line in f:
                    [self.max_sentence_len, self.max_tags_num] = [int(v) for v in line.split('\t')]

            self.IDs, self.sentences_str, self.sentences_words, self.tags_words, self.labels, _, _ = read_data(dataset_file)

        self.OOV_ID = self.word_vocab.get_id(load_data.OOV)
        self.NAME_ID = self.word_vocab.get_id(load_data.NAME)
        # self.gigaW2vEmbed, self.lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)
        # self.word_vocab_embed = self.create_word_vocab_embed()

    # def sanitise_and_lookup_embedding(self, word_id):
    #     word = self.word_vocab.get_word(word_id)

    #     if word in self.lookupGiga:
    #         word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga[word]])
    #     else:
    #         word_embed = Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<unk>"]])

    #     return word_embed

    # def create_word_vocab_embed(self):

    #     word_vocab_embed = list()

    #     # leave last word = "@PADDING"
    #     for word_id in range(0, self.word_vocab.size() - 1):
    #         word_embed = self.sanitise_and_lookup_embedding(word_id)
    #         word_vocab_embed.append(word_embed)

    #     # NOTE: adding the embed for @PADDING
    #     word_vocab_embed.append(Gigaword.norm(self.gigaW2vEmbed[self.lookupGiga["<pad>"]]))
    #     return np.array(word_vocab_embed).astype('float32')

    def pad_item(self, dataitem, type='sentence'):
        if (type is 'sentence'): 
            dataitem_padded = dataitem + [self.word_vocab.get_id(load_data.PAD)] * (self.max_sentence_len - len(dataitem))
        elif (type is 'tag'): 
            dataitem_padded = dataitem + [self.word_vocab.get_id(load_data.PAD)] * (self.max_tags_num - len(dataitem))

        return dataitem_padded
 
    def get_input(self):

        input_sentence = []
        input_tag = []
        for idx in range(len(self.sentences_words)):
            sentence_words_id = [self.word_vocab.get_id(w) for w in self.sentences_words[idx]]
            tags_words_id = [self.word_vocab.get_id(w) for w in self.tags_words[idx]]
        
            sentence_words_padded = self.pad_item(sentence_words_id, 'sentence') 
            tag_words_padded = self.pad_item(tags_words_id, 'tag') 

            # # Average the word-embeddings
            # sentence_embed = np.zeros(self.word_vocab_embed.shape[1])
            # for i in sentence_words_padded:
            #     sentence_embed += self.word_vocab_embed[i] 
            # sentence_embed = sentence_embed / len(sentence_words_padded)
            
            # tag_embed = np.zeros(self.word_vocab_embed.shape[1])
            # for i in tag_words_padded:
            #     tag_embed += self.word_vocab_embed[i] 
            # tag_embed = tag_embed / len(tag_words_padded)

            # concatenated = list(sentence_embed) + list(tag_embed)

            input_sentence.append(sentence_words_padded)
            input_tag.append(tag_words_padded)

        return (np.array(input_sentence), np.array(input_tag))

    def get_output(self):

        return self.labels


def twitter_lstm(n_inputs: int, n_outputs: int) -> Tuple[keras.Model, Dict]:
    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    model = Sequential()
    
    model.add(Dense(100, activation='tanh', input_dim=n_inputs))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam')

    kwargs = {'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')], 'batch_size': 32}

    return (model,kwargs)


def twitter_cnn(vocabulary_size: int, sentence_length: int, tag_num: int, n_outputs: int) -> Tuple[keras.Model, Dict]:
    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    # sentence
    model3 = Sequential()
    model3.add(Embedding(vocabulary_size, output_dim=256, input_length=sentence_length)) 
    model3.add(Conv1D(50, 3, activation='tanh'))
    model3.add(MaxPooling1D(pool_size=2))
    model3.add(Flatten())

    model5 = Sequential()
    model5.add(Embedding(vocabulary_size, output_dim=256, input_length=sentence_length)) 
    model5.add(Conv1D(50, 5, activation='tanh'))
    model5.add(MaxPooling1D(pool_size=2))
    model5.add(Flatten())

    model8 = Sequential()
    model8.add(Embedding(vocabulary_size, output_dim=256, input_length=sentence_length)) 
    model8.add(Conv1D(50, 8, activation='tanh'))
    model8.add(MaxPooling1D(pool_size=2))
    model8.add(Flatten())

    #tag
    model2 = Sequential()
    model2.add(Embedding(vocabulary_size, output_dim=256, input_length=tag_num)) 
    model2.add(Conv1D(50, 2, activation='tanh'))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Flatten())

    model4 = Sequential()
    model4.add(Embedding(vocabulary_size, output_dim=256, input_length=tag_num)) 
    model4.add(Conv1D(50, 4, activation='tanh'))
    model4.add(MaxPooling1D(pool_size=2))
    model4.add(Flatten())


    model = Sequential()
    model.add(Merge([model3, model5, model8, model2 ,model4], mode = 'concat'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam')

    kwargs = {'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')], 'batch_size': 32}

    return (model,kwargs)


def main():

    prediction = "E-C_en_pred.txt"
    with io.open(prediction, 'w', encoding='utf8') as f:
        f.write('ID'+ '\t' + 'Tweet'+ '\t' + 'anger'+ '\t' + 'anticipation'+ '\t' + 'disgust'+ '\t' + 'fear'+ '\t' + 'joy'+ '\t' + 'love'+ '\t' + 'optimism'+ '\t' + 'pessimism'+ '\t' + 'sadness'+ '\t' + 'surprise'+ '\t' + 'trust'+ '\n')
 
        train_dataset = load_data('train')
        train_sentence, train_tag = train_dataset.get_input()
        train_out = train_dataset.get_output()
        sentence_len = train_sentence.shape[1]
        tag_n = train_tag.shape[1]
        n_outputs = train_out.shape[1]

        dev_dataset = load_data('dev')
        dev_sentence, dev_tag = dev_dataset.get_input()
        dev_out = dev_dataset.get_output() 

        # request a model
        model, kwargs = twitter_cnn(train_dataset.word_vocab.size(), sentence_len, tag_n, n_outputs)
        model.fit([train_sentence, train_sentence, train_sentence, train_tag, train_tag], train_out, verbose=0, epochs=100)

        preds = model.predict([dev_sentence, dev_sentence, dev_sentence, dev_tag, dev_tag])
        preds[preds>= 0.5] = 1
        preds[preds<0.5] = 0
        for i in range(preds.shape[0]):
            prediction = ''
            for val in preds[i]:
                prediction = prediction + '\t' + str(int(val))
            f.write(dev_dataset.IDs[i] + '\t' + dev_dataset.sentences_str[i] + prediction + '\n') 


if __name__ == '__main__':
    main()
