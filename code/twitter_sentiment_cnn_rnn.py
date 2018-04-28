#!/usr/bin/env python

import numpy as np
import re, string 
from typing import Tuple, List, Dict
from vocabulary import *
from itertools import chain
from keras.models import Sequential , Model
import keras
from keras import optimizers
from keras.layers import Dense, Merge, Embedding, Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional, Input, Reshape,Convolution2D, TimeDistributed, Convolution1D, merge, LSTM, Dropout, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import pydot


def read_data(filename):
    IDs = []
    sentences_str = []
    
    char_str = []
    sentences_words = []
    tags = []
    
    max_word_len = 16
    max_tags_num = 8
    max_sentence_len = 46
    
    labels = []
    
    with open(filename) as f:
        f.readline()
        for line in f:
            vals = line.strip().split('\t')
            ID = vals[0].strip()
            
            sentence_str = vals[1].strip()
            sentence_str = sentence_str.lower()
            sentence_str = sentence_str.replace(':(', " sad ")
            sentence_str = sentence_str.replace('T_T', " sad ")
            sentence_str = sentence_str.replace(':)', " joy ")
            sentence_str = sentence_str.replace(':D', " joy ")
            sentence_str = sentence_str.replace('^^', " joy ")
            sentence_str = sentence_str.replace(';-)', " cute ")
            sentence_str = sentence_str.replace(':P', " blink ")
            sentence_str = sentence_str.replace('\\n', " ")
            sentence_str = sentence_str.replace('...'," ellipsis ")
            sentence_str = ' '.join(sentence_str.split())   #reduce multi space to one
            
            # split at each punctuation
            split_words = re.split(r'(\\n| |#|%|\'|\"|,|:|;|!|=|\.|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\/|\||\<|\>|\^|\`|\~)', sentence_str)
            sentence_words = []
            for w_idx, w in enumerate(split_words): 
                if w is '#':
                    pass
                elif w_idx > 0 and split_words[w_idx - 1] is '#':
                    sentence_words.append('#'+w)
                else:  
                    sentence_words.append(w)
            
            tag = []
            istag = max_tags_num    # conside last 8 tags at most, others word begin with '#' are considered as normal word
            sentence_words.reverse()    # when words with # locates at the end of sentence, they are considered as tags
            
            i = 0
            while i < len(sentence_words):
                # for i, word in enumerate(sentence_words):
                word = sentence_words[i]
                
                if len(word) == 0 or word is ' ':
                    sentence_words.remove(word)
                    i -= 1

                elif word[0] is '@':
                    sentence_words[i] = "@NID"       #someone's Network ID
                    istag = 0    
                
                else:

                    if word[0] is '#' and istag > 0:    #tags
                        for t in word.split('#'):
                            if len(t) > 0 and len(t) <= max_word_len: 
                                tag.append(t)
                            if len(t) > 0 and len(t) > max_word_len: 
                                tag.append(t[:max_word_len])
                        sentence_words.remove(word)
                        i -= 1
                        istag -= 1

                    else:
                        
                        if word[0] is '#' and istag == 0:    #words in sentence
                            word = word[1:]
                            sentence_words[i] = word
                        
                        if word.isdigit():
                            sentence_words[i] = "num"
                            istag = 0
                        
                        if any(char.isdigit() for char in word) and any(char.isalpha() for char in word) and word.isalnum():
                            sentence_words[i] = "xnumx"
                            istag = 0
                    
                        if not word.isalnum():
                            if len(word) > 1:
                                l_idx = 0
                                while l_idx < len(word):
                                    
                                    if not word[l_idx].isalnum() and word[l_idx] is not ' ':
                                        if l_idx > 0 and word[l_idx] in word[:l_idx] and l_idx < len(word)-1:
                                            word = word[:l_idx] + ' ' + word[l_idx+1:]
                                        elif l_idx > 0 and word[l_idx] in word[:l_idx] and l_idx == len(word)-1:
                                            word = word[:l_idx] + ' ' 
                                        
                                        elif l_idx > 0 and l_idx < len(word)-1:
                                            word = word[:l_idx] + ' ' + word[l_idx] + ' ' + word[l_idx+1:]
                                            l_idx += 1
                                            # print ('yesssss')
                                            # print (word)
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

                        # if max_word_len > 30:
                        #     print (sentence_words[i])
                        #     sys.exit()
                        if len(sentence_words[i]) > max_word_len:
                            sentence_words[i] = sentence_words[i][:max_word_len]

                i += 1
            

            IDs.append(ID)
            sentences_str.append(vals[1].strip())
            sentence_words.reverse()
            if len(sentence_words) <= max_sentence_len: 
                sentences_words.append(sentence_words)
            else:
                sentences_words.append(sentence_words[:max_sentence_len])
            tags.append(tag)

            labels.append(vals[2:13])
         
        labels = np.array(labels)
             
    # print('#########')           
    # print(IDs)
    # print(sentences_str)  
    # print(sentences_words)  
    # print(tags)        
    # # print(labels)  
    # print(max_word_len)  
    # print(max_sentence_len)  
    # print(max_tags_num)  
    # sys.exit()
    return IDs, sentences_str, sentences_words, tags, labels, max_word_len, max_sentence_len, max_tags_num


class data_preprocess():
    word_PAD = "@PADDING"
    char_PAD = '-'
    NID = "@NID"   # some's network ID
    NID_ID = -1
    OOV = "</s>"
    OOV_ID = -2
    

    def __init__(self, type=' '):

        dataset_file = "2018-E-c-En-" + type + ".txt" 
        self.IDs, self.sentences_str, self.sentences_words, self.tags_words, self.labels, self.max_word_len, self.max_sentence_len, self.max_tags_num = read_data(dataset_file)

        if type is 'train':
            # self.IDs, self.sentences_str, self.sentences_words, self.tags_words, self.labels, self.max_word_len, self.max_sentence_len, self.max_tags_num = read_data(dataset_file)

            #word_vocabulary
            self.word_vocab = Vocabulary()
            self.word_vocab.add("@PADDING", 0)
            for word in chain.from_iterable(zip(*self.sentences_words)):
                self.word_vocab.add(word)
            for word in chain.from_iterable(zip(*self.tags_words)):
                self.word_vocab.add(word)
            self.word_vocab.add(self.OOV, 0)

            word_vocab_file = "word_vocabulary_train.txt"
            self.word_vocab.to_file(word_vocab_file)


            #character_vocabulary
            self.char_vocab = Vocabulary()
            self.char_vocab.add("-", 0)    #padding
            for c in chain.from_iterable(zip(*self.sentences_str)):
                self.char_vocab.add(c)
            self.char_vocab.add(self.OOV, 0)     

            char_vocab_file = "char_vocabulary_train.txt"
            self.char_vocab.to_file(char_vocab_file)
 

        elif type is 'dev':
            
            word_vocab_file = "word_vocabulary_train.txt"
            self.word_vocab = Vocabulary.from_file(word_vocab_file)

            char_vocab_file = "char_vocabulary_train.txt"
            self.char_vocab = Vocabulary.from_file(char_vocab_file)

        self.OOV_ID = self.word_vocab.get_id(data_preprocess.OOV)
        self.NID_ID = self.word_vocab.get_id(data_preprocess.NID)
      
    def pad_item(self, dataitem, type=' '):
        if (type is 'char'): 
            words_item = []
            for item in dataitem:    #item: char ids in each word; dataitem: char ids of words in a line
                word_padded = item + [self.char_vocab.get_id(data_preprocess.char_PAD)] * (self.max_word_len - len(item))   
                words_item.append(word_padded)
            dataitem_padded = words_item + [[self.char_vocab.get_id(data_preprocess.char_PAD)] * self.max_word_len] * (self.max_sentence_len - len(dataitem))

        elif (type is 'sentence'): 
            dataitem_padded = dataitem + [self.word_vocab.get_id(data_preprocess.word_PAD)] * (self.max_sentence_len - len(dataitem))
        elif (type is 'tag'): 
            dataitem_padded = dataitem + [self.word_vocab.get_id(data_preprocess.word_PAD)] * (self.max_tags_num - len(dataitem))

        return dataitem_padded
 
    def get_input(self):
 
        chars_id = [[[self.char_vocab.get_id(char) for char in w] for w in sentence_words] for sentence_words in self.sentences_words]
        chars_id_padded = [self.pad_item(line_chars, 'char') for line_chars in chars_id]

        sentences_words_id = [[self.word_vocab.get_id(w) for w in sentence_words] for sentence_words in self.sentences_words]
        sentences_words_id_padded = [self.pad_item(sentence_words_id, 'sentence') for sentence_words_id in sentences_words_id]

        tags_words_id = [[self.word_vocab.get_id(t) for t in tag_words] for tag_words in self.tags_words]
        tags_words_id_padded = [self.pad_item(tag_words_id, 'tag') for tag_words_id in tags_words_id]

        return (np.array(chars_id_padded), np.array(sentences_words_id_padded), np.array(tags_words_id_padded))

    def get_output(self):

        return self.labels


def twitter_rnn(char_vocabulary_size: int, word_vocabulary_size: int, word_len: int, sentence_length: int, tag_num: int, n_outputs: int) -> Tuple[keras.Model, Dict]:
    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    # characters
    model_char = Sequential()
    model_char.add(Embedding(char_vocabulary_size, output_dim=30, input_length=word_len)) 
    model_char.add(Bidirectional(GRU(50, return_sequences=True))) 
    model_char.add(Bidirectional(GRU(50)))  

    # sentence
    model_sentence = Sequential()
    model_sentence.add(Embedding(word_vocabulary_size, output_dim=256, input_length=sentence_length)) 
    model_sentence.add(Bidirectional(GRU(128, return_sequences=True))) 
    model_sentence.add(Bidirectional(GRU(50)))  

    #tag
    model_tag = Sequential()
    model_tag.add(Embedding(word_vocabulary_size, output_dim=256, input_length=tag_num)) 
    model_tag.add(Flatten())  
    model_tag.add(Dense(100, activation='tanh')) 


    model = Sequential()
    model.add(Merge([model_char, model_sentence, model_tag], mode = 'concat'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

    print(model.summary())
    plot_model(model, show_shapes = True, to_file='rnn10.png')

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

def twitter_cnn_rnn(char_vocabulary_size: int, word_vocabulary_size: int, word_len: int, sentence_length: int, tag_num: int, n_outputs: int) -> Tuple[keras.Model, Dict]:

    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    __emb_dim = 80
    __char_emb_dim = 30
    # batch_size = 32
    n_filters = 50
    input1 = Input(shape=(sentence_length, word_len))
    input2 = Input(shape=(sentence_length,))
    
    word_embedding = Embedding(input_dim=word_vocabulary_size, output_dim = __emb_dim, input_length=sentence_length)(input2)
    char_embedding = TimeDistributed(Embedding(input_dim=char_vocabulary_size, output_dim = __char_emb_dim), batch_input_shape=(sentence_length, word_len))(input1) 
    char_cnn1 = TimeDistributed(Convolution1D(n_filters, 2, activation='relu', border_mode='same'))(char_embedding) 
    char_max_pool = TimeDistributed(MaxPooling1D((word_len)))(char_cnn1) 
    flat = TimeDistributed(Flatten())(char_max_pool)

    concat = merge([word_embedding, flat], mode='concat')
    blstm = Bidirectional(LSTM(output_dim=80, init='uniform', inner_init='uniform', forget_bias_init='one', return_sequences=True, activation='tanh', inner_activation='sigmoid'), merge_mode='sum')(concat)
    dropper = Dropout(0.2)(blstm)
    # dense = TimeDistributed(Dense(n_outputs, activation='sigmoid'))(dropper)
    # 
    avgpool =  GlobalAveragePooling1D()(dropper) 
    dense = Dense(n_outputs, activation='sigmoid')(avgpool)
    model = Model(inputs=[input1, input2], outputs=dense)

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy']) 
    # # characters
    # model_char = Sequential()
    # model_char.add(Embedding(char_vocabulary_size, output_dim=8, input_length=word_len)) 
    # model_char.add(Conv1D(100, 5, activation='tanh'))
    # model_char.add(MaxPooling1D(pool_size=2))
    # model_char.add(Conv1D(50, 3, activation='tanh')) 
    # model_char.add(MaxPooling1D(pool_size=2))
    # model_char.add(Flatten()) 
    # model_char.add(Dense(100, activation='tanh'))

    # # sentence
    # model_sentence = Sequential()
    # model_sentence.add(Embedding(word_vocabulary_size, output_dim=256, input_length=sentence_length)) 
    # model_sentence.add(Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.2))) 
    # model_sentence.add(Bidirectional(GRU(50)))  

    # #tag
    # model_tag = Sequential()
    # model_tag.add(Embedding(word_vocabulary_size, output_dim=256, input_length=tag_num)) 
    # model_tag.add(Flatten())  
    # model_tag.add(Dense(500, activation='tanh')) 
    # model_tag.add(Dense(100, activation='tanh')) 

    # model = Sequential()
    # model.add(Merge([model_char, model_sentence, model_tag], mode = 'concat'))
    # model.add(Dense(n_outputs, activation='sigmoid'))
    

    print(model.summary())
    plot_model(model, show_shapes = True, to_file='cnn_rnn.png')

    kwargs = {'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')], 'batch_size': 32}

    return (model,kwargs)


def main():

    prediction = "E-C_en_pred.txt"
    with io.open(prediction, 'w', encoding='utf8') as f:
        f.write('ID'+ '\t' + 'Tweet'+ '\t' + 'anger'+ '\t' + 'anticipation'+ '\t' + 'disgust'+ '\t' + 'fear'+ '\t' + 'joy'+ '\t' + 'love'+ '\t' + 'optimism'+ '\t' + 'pessimism'+ '\t' + 'sadness'+ '\t' + 'surprise'+ '\t' + 'trust'+ '\n')
 
        train_dataset = data_preprocess('train')
        train_char, train_sentence, train_tag = train_dataset.get_input()
        train_out = train_dataset.get_output()

        char_len = train_char.shape[-1]
        sentence_len = train_sentence.shape[1]
        tag_n = train_tag.shape[1]
        n_outputs = train_out.shape[1]

        dev_dataset = data_preprocess('dev')
        dev_char, dev_sentence, dev_tag = dev_dataset.get_input()
        dev_out = dev_dataset.get_output() 

        # request a model
        # model, kwargs = twitter_cnn(train_dataset.word_vocab.size(), n_inputs, n_outputs)
        # model.fit(train_concate, train_out, verbose=0, epochs=100)
        # preds = model.predict(dev_concate) 

        model, kwargs = twitter_cnn_rnn(train_dataset.char_vocab.size(), train_dataset.word_vocab.size(), char_len, sentence_len, tag_n, n_outputs)
        # model.fit([train_char, train_sentence, train_tag], train_out, verbose=0, epochs=100)
        model.fit([train_char, train_sentence], train_out, verbose=0, epochs=100)
        preds = model.predict([dev_char, dev_sentence])
        preds[preds>= 0.5] = 1
        preds[preds<0.5] = 0
        for i in range(preds.shape[0]):
            prediction = ''
            for val in preds[i]:
                prediction = prediction + '\t' + str(int(val))
            f.write(dev_dataset.IDs[i] + '\t' + dev_dataset.sentences_str[i] + prediction + '\n') 


if __name__ == '__main__':
    main()
