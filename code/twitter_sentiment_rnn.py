#!/usr/bin/env python

import numpy as np
import re, string 
from typing import Tuple, List, Dict
from vocabulary import *
from itertools import chain
from keras.models import Sequential 
import keras.optimizers
from keras.layers import Dense, Merge, Embedding, Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import pydot

def read_data(filename):
    IDs = []
    sentences_str = []
    
    char_str = []
    sentences_words = []
    tags = []
    
    max_word_len = 0
    max_tags_num = 0
    max_sentence_len = 0
    
    labels = []
    
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
            sentence_str = sentence_str.replace('\\n', " ")
            sentence_str = ' '.join(sentence_str.split())
            
            # split at each punctuation
            split_words = re.split(r'(\\n|%|\'|\"|,|:|;|!|=|\.|\(|\)|\$|\?|\*|\+|\-|\]|\[|\{|\}|\\|\/|\||\<|\>|\^|\`|\~)', sentence_str)
            sentence_words = []
            for words in split_words:
                sentence_words.extend(words.split())
            
            tag = []
            istag = 1
            sentence_words.reverse()    # when words with # locates at the end of sentence, they are considered as tags
            
            i = 0
            while i < len(sentence_words):
                # for i, word in enumerate(sentence_words):
                word = sentence_words[i]
                
                if len(word) == 0:
                    sentence_words.remove(word)
                
                elif word[0] is '#' and istag == 1:    #tags
                    tag.append(word[1:])
                    sentence_words.remove(word)
                
                else:
                    if word[0] is '#' and istag == 0:    #words in sentence
                        sentence_words[i] = word[1:]
                    
                    elif word[0] is '@':
                        sentence_words[i] = "@NID"       #someone's Network ID
                        istag = 0
                    
                    elif word.isdigit():
                        sentence_words[i] = "num"
                        istag = 0
                    
                    elif any(char.isdigit() for char in word) and word.isalnum():
                        sentence_words[i] = "xnumx"
                        istag = 0
                
                    elif not word.isalnum():
                        if len(word) > 1:
                            l_idx = 0
                            while l_idx < len(word):
                                
                                if not word[l_idx].isalnum() and word[l_idx] is not ' ':
                                    if l_idx > 0 and word[l_idx] in word[:l_idx] and l_idx < len(word)-1:
                                        word = word[:l_idx] + ' ' + word[l_idx+1:]
                                    elif l_idx > 0 and word[l_idx] in word[:l_idx] and l_idx == len(word)-1:
                                        word = word[:l_idx] + ' ' 
                                    
                                    # elif l_idx > 0 and word[l_idx] == word[l_idx - 1] and l_idx == len(word)-1:
                                    #     word = word[:l_idx]
                                    
                                    elif l_idx > 0 and l_idx < len(word)-1:
                                        word = word[:l_idx] + ' ' + word[l_idx] + ' ' + word[l_idx+1:]
                                        l_idx += 1
                                    elif l_idx > 0 and l_idx == len(word)-1:
                                        word = word[:l_idx] + ' ' + word[l_idx]
                                        l_idx += 1
                                    elif l_idx == 0: 
                                        word = word[l_idx] + ' ' + word[l_idx+1:]
                        
                                l_idx += 1
                            
                            word = ' '.join(word.split())
                            word_split = word.split()
                            word_split.reverse()

                            if (i < len(sentence_words)-1):
                                sentence_words = sentence_words[:i] + word_split + sentence_words[i+1:]
                            else:
                                sentence_words = sentence_words[:i] + word_split
                            
                            i -= 1
    
                        istag = 0

                    else:
                        istag = 0
                      

                    if len(sentence_words[i]) > max_word_len:
                        max_word_len = len(sentence_words[i])

                    i += 1
            # if max_word_len > 70:
            #     print (word)
            #     sys.exit()
            

            IDs.append(ID)
            sentences_str.append(sentence_str)
            sentence_words.reverse()
            sentences_words.append(sentence_words)
            tags.append(tag)

            labels.append(vals[2:13])
            
            if len(tag) > max_tags_num:
                max_tags_num = len(tag)
            
            if len(sentence_words) > max_sentence_len:
                max_sentence_len = len(sentence_words)
                    
        labels = np.array(labels)
             
    # print('#########')           
    # print(IDs)
    # print(sentences_str)  
    # print(sentences_words)  
    # print(tags)        
    # print(labels)  
    # print(max_word_len)  
    # print(max_sentence_len)  
    # print(max_tags_num)  

    # sys.exit()
    return IDs, sentences_str, sentences_words, tags, labels, max_word_len, max_sentence_len, max_tags_num


class load_data():
    PAD = "@PADDING"
    OOV = "</s>"
    NID = "@NID" 
    OOV_ID = 0
    NID_ID = -1

    def __init__(self, type='train'):

        dataset_file = "2018-E-c-En-" + type + ".txt"
        # w2vfile = "vectors.goldbergdeps.txt"

        if type is 'train':
            _, _, self.sentences_words, self.tags_words, self.labels, self.max_word_len, self.max_sentence_len, self.max_tags_num = read_data(dataset_file)

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
                f.write(str(self.max_word_len) + '\t' + str(self.max_sentence_len) + '\t' + str(self.max_tags_num))

        else:

            vocab_file = "vocabulary_train.txt"
            self.word_vocab = Vocabulary.from_file(vocab_file)

            maxlen_file = "maxlen_train.txt"
            with io.open(maxlen_file, encoding='utf8') as f:
                for line in f:
                    [self.max_word_len, self.max_sentence_len, self.max_tags_num] = [int(v) for v in line.split('\t')]

            self.IDs, self.sentences_str, self.sentences_words, self.tags_words, self.labels, _, _, _ = read_data(dataset_file)

        self.OOV_ID = self.word_vocab.get_id(load_data.OOV)
        self.NID_ID = self.word_vocab.get_id(load_data.NID)

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


            input_sentence.append(sentence_words_padded)
            input_tag.append(tag_words_padded)

        return (np.array(input_sentence), np.array(input_tag))

    def get_output(self):

        return self.labels


def twitter_rnn(vocabulary_size: int, sentence_length: int, tag_num: int, n_outputs: int) -> Tuple[keras.Model, Dict]:
    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    # sentence
    model_sentence = Sequential()
    model_sentence.add(Embedding(vocabulary_size, output_dim=256, input_length=sentence_length)) 
    model_sentence.add(Bidirectional(GRU(128, return_sequences=True))) 
    model_sentence.add(Bidirectional(GRU(50)))  

    #tag
    model_tag = Sequential()
    model_tag.add(Embedding(vocabulary_size, output_dim=256, input_length=tag_num)) 
    model_tag.add(Flatten())  
    model_tag.add(Dense(100, activation='tanh')) 


    model = Sequential()
    model.add(Merge([model_sentence, model_tag], mode = 'concat'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, show_shapes = True, to_file='rnn9.png')

    kwargs = {'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')], 'batch_size': 32}

    return (model,kwargs)


def twitter_cnn(vocabulary_size: int, n_inputs: int, n_outputs: int) -> Tuple[keras.Model, Dict]:
    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    # sentence
    model = Sequential()
    model.add(Embedding(vocabulary_size, output_dim=256, input_length=n_inputs)) 
    model.add(Conv1D(100, 3, activation='tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(50, 3, activation='tanh')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(20, 3, activation='tanh')) 
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten()) 
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, show_shapes = True, to_file='cnn.png')

    kwargs = {'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')], 'batch_size': 32}

    return (model,kwargs)



def main():

    prediction = "E-C_en_pred.txt"
    with io.open(prediction, 'w', encoding='utf8') as f:
        f.write('ID'+ '\t' + 'Tweet'+ '\t' + 'anger'+ '\t' + 'anticipation'+ '\t' + 'disgust'+ '\t' + 'fear'+ '\t' + 'joy'+ '\t' + 'love'+ '\t' + 'optimism'+ '\t' + 'pessimism'+ '\t' + 'sadness'+ '\t' + 'surprise'+ '\t' + 'trust'+ '\n')
 
        train_dataset = load_data('train')
        train_sentence, train_tag = train_dataset.get_input()
        train_concate = np.hstack((train_sentence, train_tag))
        train_out = train_dataset.get_output()

        sentence_len = train_sentence.shape[1]
        tag_n = train_tag.shape[1]
        n_outputs = train_out.shape[1]
        n_inputs = train_concate.shape[1]

        # print('nums:')
        # print (sentence_len)
        # print (tag_n)
        # print (n_inputs)

        dev_dataset = load_data('dev')
        dev_sentence, dev_tag = dev_dataset.get_input()
        dev_concate = np.hstack((dev_sentence, dev_tag))
        dev_out = dev_dataset.get_output() 

        # request a model
        # model, kwargs = twitter_cnn(train_dataset.word_vocab.size(), n_inputs, n_outputs)
        # model.fit(train_concate, train_out, verbose=0, epochs=100)
        # preds = model.predict(dev_concate)
  

        model, kwargs = twitter_rnn(train_dataset.word_vocab.size(), sentence_len, tag_n, n_outputs)
        model.fit([train_sentence, train_tag], train_out, verbose=0, epochs=100)
        preds = model.predict([dev_sentence, dev_tag])
        preds[preds>= 0.5] = 1
        preds[preds<0.5] = 0
        for i in range(preds.shape[0]):
            prediction = ''
            for val in preds[i]:
                prediction = prediction + '\t' + str(int(val))
            f.write(dev_dataset.IDs[i] + '\t' + dev_dataset.sentences_str[i] + prediction + '\n') 


if __name__ == '__main__':
    main()
