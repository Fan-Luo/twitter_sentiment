#!/usr/bin/env python

import numpy as np
import re, string 
from typing import Tuple, List, Dict
from vocabulary import *
from itertools import chain
from keras.models import Sequential , Model
import keras
from keras import optimizers
from keras.layers import Dense, Merge, Embedding, Conv1D, MaxPooling1D, Flatten, GRU, Bidirectional, Input, Reshape,Convolution2D, TimeDistributed, Convolution1D, merge, LSTM, Dropout, GlobalAveragePooling1D, Lambda
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import pydot
from math import ceil
from keras.wrappers.scikit_learn import KerasClassifier
# import sklearn.metrics  
from sklearn.metrics import jaccard_similarity_score, make_scorer
from sklearn.model_selection import GridSearchCV
import tensorflow
import keras.backend as K
from keras.models import load_model, model_from_json

def read_data(filename):
    IDs = []
    sentences_str = []
    
    char_str = []
    sentences_words = []
    
    max_word_len = 16
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
            sentence_str = ' '.join(sentence_str.split())
            
            # split at each punctuation
            sentence_words = re.split(r'(\\n| |#|%|\'|\"|,|:|;|!|=|\.|\(|\)|\$|\?|\*|\+|\]|\[|\{|\}|\\|\/|\||\<|\>|\^|\`|\~)', sentence_str)             
            sentence_words.reverse()     
            
            i = 0
            while i < len(sentence_words):
                # for i, word in enumerate(sentence_words):
                word = sentence_words[i]
                
                if len(word) == 0 or word is ' ':
                    sentence_words.remove(word)
                    i -= 1

                elif word[0] is '@':
                    sentence_words[i] = "@NID"       #someone's Network ID  
                
                else:

                    if word.isdigit():
                        sentence_words[i] = "num"
                    
                    if any(char.isdigit() for char in word) and any(char.isalpha() for char in word) and word.isalnum():
                        sentence_words[i] = "xnumx"
                
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

            labels.append(vals[2:13])
         
        labels = np.array(labels)
             
    # print('#########')           
    # print(IDs)
    # print(sentences_str)  
    # print(sentences_words)      
    # # print(labels)  
    # print(max_word_len)  
    # print(max_sentence_len)  
    # sys.exit()
    return IDs, sentences_str, sentences_words, labels, max_word_len, max_sentence_len 


class data_preprocess():
    word_PAD = "@PADDING"
    char_PAD = '-'
    NID = "@NID"   # some's network ID
    NID_ID = -1
    OOV = "</s>"
    OOV_ID = -2
    

    def __init__(self, type=' '):

        dataset_file = "2018-E-c-En-" + type + ".txt" 
        self.IDs, self.sentences_str, self.sentences_words, self.labels, self.max_word_len, self.max_sentence_len = read_data(dataset_file)

        if type is 'train':
            # self.IDs, self.sentences_str, self.sentences_words, self.tags_words, self.labels, self.max_word_len, self.max_sentence_len, self.max_tags_num = read_data(dataset_file)

            #word_vocabulary
            self.word_vocab = Vocabulary()
            self.word_vocab.add("@PADDING", 0)
            for word in chain.from_iterable(zip(*self.sentences_words)):
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

        return dataitem_padded
 
    def get_input(self):
 
        chars_id = [[[self.char_vocab.get_id(char) for char in w] for w in sentence_words] for sentence_words in self.sentences_words]
        chars_id_padded = [self.pad_item(line_chars, 'char') for line_chars in chars_id]

        sentences_words_id = [[self.word_vocab.get_id(w) for w in sentence_words] for sentence_words in self.sentences_words]
        sentences_words_id_padded = [self.pad_item(sentence_words_id, 'sentence') for sentence_words_id in sentences_words_id]

        return (np.array(chars_id_padded), np.array(sentences_words_id_padded))

    def get_output(self):

        return self.labels

def lambda_fun1(x) : 
    split1, split2 = tensorflow.split(x, [46*16, 46], 1)
    return split1

def lambda_fun2(x) : 
    split1, split2 = tensorflow.split(x, [46*16, 46], 1)
    return split2

def twitter_rnn(char_vocabulary_size: int, word_vocabulary_size: int, word_len: int, sentence_length: int, tag_num: int, n_outputs: int) -> keras.Model: 
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

    return model 


def twitter_cnn(vocabulary_size: int, n_inputs: int, n_outputs: int) -> keras.Model:
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

    return model 

def twitter_cnn_rnn(char_vocabulary_size: int, word_vocabulary_size: int, word_len: int, sentence_length: int, n_outputs: int) -> keras.Model:

    """
    The neural networks will be asked to predict the 0 or more tags 

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models. 
    """

    __emb_dim = 200
    __char_emb_dim = 30
    # batch_size = 32
    n_filters = 200
    inputs = Input(shape=(sentence_length * word_len + sentence_length ,))
    # input1 = inputs[sentence_length:]
    # input2 = inputs[:sentence_length]
    input1 = Lambda(lambda_fun1)(inputs) 
    input2 = Lambda(lambda_fun2)(inputs) 

    char_embedding = Embedding(input_dim=char_vocabulary_size, output_dim = __char_emb_dim)(input1) 
    char_cnn1 = Convolution1D(n_filters, word_len, strides=word_len, activation='relu', padding='valid')(char_embedding) 
    
    word_embedding = Embedding(input_dim=word_vocabulary_size, output_dim = __emb_dim, input_length=sentence_length)(input2)

    concat1 = merge([word_embedding, char_cnn1], mode='concat')
    dropper1 = Dropout(0.2)(concat1)
    blstm = Bidirectional(LSTM(return_sequences=True, activation="tanh", unit_forget_bias=True, units=80, kernel_initializer="uniform", recurrent_initializer="uniform", recurrent_activation="sigmoid"), merge_mode='sum')(dropper1)
    # dropper2 = Dropout(0.2)(blstm)
    # dense = TimeDistributed(Dense(n_outputs, activation='sigmoid'))(dropper)
    # 
    avgpool1 =  GlobalAveragePooling1D()(blstm) 
    dropper2 = Dropout(0.1)(avgpool1)
    dense = Dense(n_outputs, activation='sigmoid')(dropper2)

    model = Model(inputs=inputs, outputs=dense)
    model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])  
    
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
    plot_model(model, show_shapes = True, to_file='cnn_rnn63.png')

    return model


def main():

    prediction = "E-C_en_pred.txt"
    with io.open(prediction, 'w', encoding='utf8') as f:
        f.write('ID'+ '\t' + 'Tweet'+ '\t' + 'anger'+ '\t' + 'anticipation'+ '\t' + 'disgust'+ '\t' + 'fear'+ '\t' + 'joy'+ '\t' + 'love'+ '\t' + 'optimism'+ '\t' + 'pessimism'+ '\t' + 'sadness'+ '\t' + 'surprise'+ '\t' + 'trust'+ '\n')
 
        train_dataset = data_preprocess('train')
        train_char, train_sentence = train_dataset.get_input()
        train_out = train_dataset.get_output()

        char_len = train_char.shape[-1]
        sentence_len = train_sentence.shape[1] 
        n_outputs = train_out.shape[1]

        #use 0.1 as validation dataset
        n_sample = train_char.shape[0]
        n_val = ceil(n_sample * 0.1)
        
        val_char, val_sentence = train_char[:n_val], train_sentence[:n_val]
        val_char = val_char.reshape((n_val,-1))
        val_in = np.hstack((val_char, val_sentence))
        val_out = train_out[:n_val]

        train_char, train_sentence = train_char[n_val:] , train_sentence[n_val:]
        train_char = train_char.reshape((n_sample-n_val,-1))
        train_in = np.hstack((train_char, train_sentence))
        train_out = train_out[n_val:]

        dev_dataset = data_preprocess('dev')
        dev_char, dev_sentence = dev_dataset.get_input()
        dev_char = dev_char.reshape((dev_char.shape[0],-1))
        dev_in = np.hstack((dev_char, dev_sentence))
        dev_out = dev_dataset.get_output() 

        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)

        # request a model
        model = twitter_cnn_rnn(train_dataset.char_vocab.size(), train_dataset.word_vocab.size(), char_len, sentence_len, n_outputs)
        kwargs = {'x': train_in, 'y': train_out, 'validation_data': (val_in, val_out), 'verbose':0, 'epochs':100, 'callbacks': [EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')], 'batch_size': 32}
        model.fit(**kwargs) 
        # # save model to disk
        model.save("model63.h5")

        # load model
        # model = load_model('model63.h5')
        # model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])  

        preds = model.predict(dev_in) 
  
        preds[preds>= 0.2] = 1
        preds[preds<0.2] = 0

        n_dev = preds.shape[0]
        sc = np.zeros(n_dev)
        sc_union = np.zeros(n_dev)
        sum_sc = 0
        
        for i in range(n_dev):
            prediction = ''
            for j,val in enumerate(preds[i]):
                prediction = prediction + '\t' + str(int(val))

                if(int(preds[i][j]) == 1 and int(preds[i][j]) == int(dev_out[i][j])):
                    sc[i] += 1
                if(int(preds[i][j])):
                    sc_union[i] += 1
                if(int(dev_out[i][j])):
                    sc_union[i] += 1

            f.write(dev_dataset.IDs[i] + '\t' + dev_dataset.sentences_str[i] + prediction + '\n') 
            
            sc_union[i] -= sc[i]       
            if sc_union[i] > 0:
                sum_sc += sc[i] * 1.0 / sc_union[i]
            else: 
                sum_sc += 1
            
        jaccard_score = sum_sc / (1.0 * n_dev)

        print('jaccard_similarity_score:'+str(jaccard_score))
        with io.open('result.txt', 'w', encoding='utf8') as fo:
            fo.write('jaccard_similarity_score:'+str(jaccard_score)) 
   


if __name__ == '__main__':
    main()
