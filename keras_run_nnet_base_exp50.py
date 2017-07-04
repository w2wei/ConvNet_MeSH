'''only use query and answer text, no additional features such as overlap and knn counts'''

from datetime import datetime
from sklearn import metrics
from theano import tensor as T
import cPickle
import os
import sys
import theano
import time
from collections import defaultdict
import pandas as pd
from copy import copy

import warnings
warnings.filterwarnings("ignore")  # TODO remove

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'

## keras settings
import numpy as np 
np.random.seed(1337)
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D, GlobalMaxPooling1D, Convolution2D, MaxPooling2D
from keras import backend as K 
from keras.utils import np_utils
from keras.models import load_model

## end of keras settings
def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score

def map_score(qids, labels, preds):
  '''evaluate MAP scores'''
  qid2cand = defaultdict(list)
  for qid, label, pred in zip(qids, labels, preds):
    qid2cand[qid].append((pred, label))

  average_precs = []
  for qid, candidates in qid2cand.iteritems():
    average_prec = 0
    running_correct_count = 0
    for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
      if label > 0:
        running_correct_count += 1
        average_prec += float(running_correct_count) / i
    average_precs.append(average_prec / (running_correct_count + 1e-6))
  map_score = sum(average_precs) / len(average_precs)
  return map_score

def main():
  t0 = time.time()
  exp_num = sys.argv[1]
  # ZEROUT_DUMMY_WORD = False
  ZEROUT_DUMMY_WORD = True

  ## Load data
  # data_dir = "/home/w2wei/projects/pointwiseLTR/data/sample/SAMPLE_TRAIN"
  base_dir = "/data/projects/pointwiseLTR/data/knn_sample"
  data_dir = os.path.join(base_dir, sys.argv[1])  

  # data_dir = 'TRAIN'

  q_train = np.load(os.path.join(data_dir, 'train.questions.npy'))
  a_train = np.load(os.path.join(data_dir, 'train.answers.npy'))
  y_train = np.load(os.path.join(data_dir, 'train.labels.npy'))

  q_dev = np.load(os.path.join(data_dir, 'dev.questions.npy'))
  a_dev = np.load(os.path.join(data_dir, 'dev.answers.npy'))
  y_dev = np.load(os.path.join(data_dir, 'dev.labels.npy'))
  qids_dev = np.load(os.path.join(data_dir, 'dev.qids.npy'))

  q_test = np.load(os.path.join(data_dir, 'test.questions.npy'))
  a_test = np.load(os.path.join(data_dir, 'test.answers.npy'))
  y_test = np.load(os.path.join(data_dir, 'test.labels.npy'))
  qids_test = np.load(os.path.join(data_dir, 'test.qids.npy'))

  q_test = q_test[:2000]
  a_test = a_test[:2000]
  y_test = y_test[:2000]
  qids_test = qids_test[:2000]
  orig_y_test = copy(y_test)

  print 'y_train', np.unique(y_train, return_counts=True)
  print 'y_dev', np.unique(y_dev, return_counts=True)
  print 'y_test', np.unique(y_test, return_counts=True)

  print 'y_train', y_train.shape
  print 'y_dev', y_dev.shape
  print 'y_test', y_test.shape

  print 'q_train', q_train.shape
  print 'q_dev', q_dev.shape
  print 'q_test', q_test.shape

  print 'a_train', a_train.shape
  print 'a_dev', a_dev.shape
  print 'a_test', a_test.shape

  t1 = time.time()
  print "Loading data: ", t1-t0

  np_rng = np.random.RandomState(123)
  q_max_sent_size = q_train.shape[1]
  a_max_sent_size = a_train.shape[1]
  print 'max', np.max(a_train)
  print 'min', np.min(a_train)
  print "q_max_sent_size: ", q_max_sent_size
  print "a_max_sent_size: ", a_max_sent_size

  # Load word2vec embeddings
  ndim = 100 ## dim of a word vec, fixed in this exp
  print "Generating random emb for the vocabulary. vocab dim %d"%ndim
  fname_vocab = os.path.join(data_dir, 'vocab.pickle')
  alphabet = cPickle.load(open(fname_vocab))
  random_words_count = 0
  vocab_emb = np.zeros((len(alphabet) + 1, ndim))
  for word, idx in alphabet.iteritems():
      word_vec = np.random.uniform(-0.25, 0.25, ndim)
      random_words_count += 1
      vocab_emb[idx] = word_vec
  print "vocab_emb shape: ", vocab_emb.shape
  outfile = os.path.join(data_dir, 'emb_vocab.npy')
  np.save(outfile, vocab_emb)
  ndim = vocab_emb.shape[1]
  dummpy_word_idx = np.max(a_train)
  print "Word embedding matrix size:", vocab_emb.shape
  print "dummy word idx ", dummpy_word_idx
  print "ndim ", ndim # 100

  ## build a keras model
  nb_out = 2 # 2 outcomes
  nb_epoch = 1
  nb_filter=100 ## nkernels
  filter_length=5 ## q_filter_widths and a_filter_widths
  batch_size = 50
  word_vocab_size = np.max(q_train)+1 # vocab size 17022, max_features
  word_embedding_dims = vocab_emb.shape[1] ## dimension of word vectors, 50
  q_maxlen = q_max_sent_size ## max length of questions (# words)
  a_maxlen = a_max_sent_size ## max length of answers (# words)
  hidden_dims=250
  dropout_rate = 0.5
  learning_rate = 0.1
  max_norm = 0
  print 'batch_size', batch_size
  print 'nb_epoch', nb_epoch
  print "vocab max", word_vocab_size 

  ## learn a model
  try:
    print 1.0/0
    final_model = load_model('exp_50_base_tanh.h5')
  except:
    print "\nBuilding a model\n#################"
    q_model = Sequential()
    q_model.add(Embedding(input_dim=word_vocab_size,output_dim=word_embedding_dims, input_length=q_maxlen,dropout=0.2))
    print "q embedding shape: ",q_model.output_shape ## None, 33, 50
    q_model.add(Convolution1D(nb_filter, filter_length, border_mode='same', activation='tanh', subsample_length=1))
    print "q conv layer 1 shape: ",q_model.output_shape ## None, 33, 100
    q_model.add(GlobalMaxPooling1D())
    print "q max pooling 1 shape: ",q_model.output_shape ## None, 33, 100

    a_model = Sequential()
    a_model.add(Embedding(input_dim=word_vocab_size,output_dim=word_embedding_dims, input_length=a_maxlen,dropout=0.2))
    print "a embedding shape: ", a_model.output_shape ## None, 40, 50
    a_model.add(Convolution1D(nb_filter, filter_length, border_mode='same', activation='tanh', subsample_length=1))
    print "a embedding shape: ", a_model.output_shape ## None, 40, 50  
    a_model.add(GlobalMaxPooling1D())
    print "a max pooling 1 shape: ",a_model.output_shape ## None, 33, 100

    ## merge conv layer outputs and overlap features
    q_a_model = Merge([q_model, a_model], mode='concat')
    print "merged q_a_model shape: ", q_a_model.output_shape

    final_model = Sequential()
    final_model.add(q_a_model)
    final_model.add(Dropout(dropout_rate))

    final_model.add(Dense(nb_filter))
    final_model.add(Activation('tanh'))
    final_model.add(Dropout(dropout_rate))
    final_model.add(Dense(nb_out))
    final_model.add(Activation('softmax'))
    
    ## training
    y_train=np_utils.to_categorical(y_train, nb_out)
    y_dev=np_utils.to_categorical(y_dev, nb_out)
    y_test=np_utils.to_categorical(y_test, nb_out)
    final_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['fmeasure'])
    t2 = time.time()
    print "building a model: ", t2-t1
    final_model.fit([q_train, a_train],y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=([q_dev, a_dev],y_dev), shuffle=True)
    final_model.save('exp_50_base_tanh.h5')
    t3 = time.time()
    print "learning time: ", t3-t2

  # prediction
  print "prediction..."
  t3 = time.time()
  ## pred is a numpy array of results
  pred = final_model.predict([q_test, a_test], verbose=0)#batch_size=batch_size
  pred_list = np.argmax(pred, axis=1)
  print "actual results"
  output = np.array([qids_test.astype(int), orig_y_test.astype(int), pred_list.astype(int)]).T
  print output
  np.savetxt("output_res.txt", output, fmt='%d', delimiter=' ', newline='\n')
  
  score = map_score(qids_test.astype(int), orig_y_test.astype(int), pred_list.astype(int))
  print "map: ", score
  prec = metrics.precision_score(orig_y_test.astype(int), pred_list.astype(int))
  print "precision: ", prec
  reca = metrics.recall_score(orig_y_test.astype(int), pred_list.astype(int))
  print "recall: ", reca
  print

  t4 = time.time()
  print "prediction time: ", t4-t3
  
  # evaluate models
  print "evaluating..."
  t2 = time.time()
  y_test=np_utils.to_categorical(orig_y_test, nb_out)
  score=final_model.evaluate([q_test, a_test], y_test, verbose=0)
  print "accuracy: ", score 
  t3 = time.time()
  print "evaluation time: ", t3-t2

  # binary cross entropy [0.42901692091043481, 0.83124588014424117]
  # categorical cross entropy [0.42435971925976879, 0.83586025061227021]
  # acc = 0.833, KL
  # acc = 0.837, MSE
  # acc = 0.813, mean_absolute_error
  # acc = 0.813, cosine_proximity

if __name__ == '__main__':
  main()
