'''
  This script works.

  how to show the input of a computation graph? for example, show the input for prediction?

  Created on August 12, 2016
  Updated on August 17, 2016
  @author: Wei Wei
'''

from sklearn import metrics
from theano import tensor as T
import cPickle
import numpy
import os
import sys
import theano
import time
from collections import defaultdict
import subprocess
import pandas as pd
from tqdm import tqdm
import nn_layers
import sgd_trainer

# import warnings
# warnings.filterwarnings("ignore")  # TODO remove
theano.config.optimizer='fast_compile'
# theano.config.floatX = 'float32'
# theano.config.mode = 'FAST_COMPILE'
# theano.config.mode = 'FAST_RUN'
# theano.config.mode = 'DebugMode'
# from theano.compile.debugmode import DebugMode
# theano.config.compute_test_value = 'warn'
# theano.config.compute_test_value = 'off'
# theano.config.exception_verbosity = 'high'
# allow_gc=False

def main():
  # mode = 'TRAIN'
  # data_dir = mode
  # q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
  # a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
  # q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
  # a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
  # y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
  # qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))

  ## Load data
  base_dir = "/home/w2wei/projects/pointwiseLTR/data/knn_sample"
  data_dir = os.path.join(base_dir, sys.argv[1])

  q_test = numpy.load(os.path.join(data_dir, 'test.questions.npy'))
  a_test = numpy.load(os.path.join(data_dir, 'test.answers.npy'))
  q_overlap_test = numpy.load(os.path.join(data_dir, 'test.q_overlap_indices.npy'))
  a_overlap_test = numpy.load(os.path.join(data_dir, 'test.a_overlap_indices.npy'))
  q_knn_count_test = numpy.load(os.path.join(data_dir, 'test.q_knn_counts.npy'))
  a_knn_count_test = numpy.load(os.path.join(data_dir, 'test.a_knn_counts.npy'))
  q_mti_test = numpy.load(os.path.join(data_dir, 'test.q_mti.npy'))
  a_mti_test = numpy.load(os.path.join(data_dir, 'test.a_mti.npy'))    
  y_test = numpy.load(os.path.join(data_dir, 'test.labels.npy'))
  qids_test = numpy.load(os.path.join(data_dir, 'test.qids.npy'))
  print "Data loaded"
  ### reconstruction method 3
  numpy_rng = numpy.random.RandomState(123)  
  # batch_size = 50
  batch_size = 5
  # nnet_outdir = "exp.out/ndim=55;batch=50;max_norm=0;learning_rate=0.1;2016-08-18-09.39.15" ## Severyn's experiment on TRAIN
  # nnet_outdir = "exp.out/Exp_30; ndim=115;batch=5;max_norm=0;learning_rate=0.1;2016-07-11-22.32.45" ## Exp_30, setting 1
  # nnet_outdir = "exp.out/Exp_30; ndim=115;batch=5;max_norm=0;learning_rate=0.1;2016-07-11-22.34.10" ## Exp_30, setting 2
  nnet_outdir = "exp.out/Exp_30; ndim=115;batch=5;max_norm=0;learning_rate=0.1;2016-07-11-22.35.06" ## Exp_30, setting 3

  nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
  train_nnet, test_nnet = cPickle.load(file(nnet_fname,'rb'))
  print "Uncompiled model loaded"
  # batch_x_q = T.lmatrix('batch_x_q')
  # batch_x_a = T.lmatrix('batch_x_a')
  # batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
  # batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')

  batch_x_q = T.lmatrix('batch_x_q')
  batch_x_a = T.lmatrix('batch_x_a')
  batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
  batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
  batch_x_q_knn_count = T.lmatrix('batch_x_q_knn_count')
  batch_x_a_knn_count = T.lmatrix('batch_x_a_knn_count')
  batch_x_q_mti = T.lmatrix('batch_x_q_mti')
  batch_x_a_mti = T.lmatrix('batch_x_a_mti')
  # batch_y = T.ivector('batch_y')

  # nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier = test_nnet.layers
  # nnet_q.set_input((batch_x_q, batch_x_q_overlap))
  # nnet_a.set_input((batch_x_a, batch_x_a_overlap))
  # pairwise_layer.set_input((nnet_q.output, nnet_a.output))
  # hidden_layer.set_input(pairwise_layer.output)
  # classifier.set_input(hidden_layer.output)

  nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier = test_nnet.layers
  nnet_q.set_input((batch_x_q, batch_x_q_overlap, batch_x_q_knn_count, batch_x_q_mti))
  nnet_a.set_input((batch_x_a, batch_x_a_overlap, batch_x_a_knn_count, batch_x_a_mti))
  pairwise_layer.set_input((nnet_q.output, nnet_a.output))
  hidden_layer.set_input(pairwise_layer.output)
  classifier.set_input(hidden_layer.output)

  predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

  # inputs_pred = [batch_x_q,
  #                batch_x_a,
  #                batch_x_q_overlap,
  #                batch_x_a_overlap,
  #                ]

  inputs_pred = [batch_x_q,
                 batch_x_a,
                 batch_x_q_overlap,
                 batch_x_a_overlap,
                 batch_x_q_knn_count,
                 batch_x_a_knn_count,
                 batch_x_q_mti,
                 batch_x_a_mti,
                 ]

  pred_prob_fn = theano.function(inputs=inputs_pred, outputs=predictions_prob)
  print "Model compiled"
  ## populate parameters
  # param_fname = os.path.join(nnet_outdir, "best_dev_params.epoch=05;batch=00010;dev_acc=79.40.dat") ## Severyn's experiment on TRAIN
  # param_fname = os.path.join(nnet_outdir, "best_dev_params.epoch=04;batch=00014;dev_acc=88.76.dat") ## Exp_30, setting 1
  # param_fname = os.path.join(nnet_outdir, "best_dev_params.epoch=04;batch=00014;dev_acc=88.05.dat") ## Exp_30, setting 2
  param_fname = os.path.join(nnet_outdir, "best_dev_params.epoch=05;batch=00014;dev_acc=86.36.dat") ## Exp_30, setting 3
  best_params = cPickle.load(file(param_fname,'rb'))  
  print "Best parameters loaded"

  params = test_nnet.params
  for i, param in enumerate(best_params):
      params[i].set_value(param, borrow=True)

  # test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, y_test], batch_size=batch_size, randomize=False)
  test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
  print "training data ready"
  # def predict_prob_batch(batch_iterator):
  #     preds = numpy.hstack([pred_prob_fn(batch_x_q, 
  #                                        batch_x_a, 
  #                                        batch_x_q_overlap, 
  #                                        batch_x_a_overlap) for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, _ in batch_iterator])
  #     return preds[:batch_iterator.n_samples]    

  def predict_prob_batch(batch_iterator):
      preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
                                         batch_x_q_knn_count, batch_x_a_knn_count, batch_x_q_mti, batch_x_a_mti) \
                                         for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
                                             batch_x_q_knn_count, batch_x_a_knn_count, \
                                             batch_x_q_mti, batch_x_a_mti, _ in batch_iterator])
      return preds[:batch_iterator.n_samples]  

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

  print "Predicting..."
  t0=time.time()
  y_pred_test = predict_prob_batch(test_set_iterator)
  t1=time.time()
  print "Prediction time: ", t1-t0
  test_acc = map_score(qids_test, y_test, y_pred_test) * 100
  print "MAP on test set: ", test_acc/100.0
  print

  # print "Running trec_eval script..."
  # N = len(y_pred_test)

  # df_submission = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
  # df_submission['qid'] = qids_test
  # df_submission['iter'] = 0
  # df_submission['docno'] = numpy.arange(N)
  # df_submission['rank'] = 0
  # df_submission['sim'] = y_pred_test
  # df_submission['run_id'] = 'nnet'
  # df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

  # df_gold = pd.DataFrame(index=numpy.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
  # df_gold['qid'] = qids_test
  # df_gold['iter'] = 0
  # df_gold['docno'] = numpy.arange(N)
  # df_gold['rel'] = y_test
  # df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

  # subprocess.call("/bin/sh run_eval.sh '{}'".format(nnet_outdir), shell=True)

if __name__ == '__main__':
  main()

  ## reconstruction method 2
  # numpy_rng = numpy.random.RandomState(123)  
  # batch_size = 50
  # nnet_outdir = "exp.out/ndim=55;batch=50;max_norm=0;learning_rate=0.1;2016-08-12-10.48.34"

  # nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
  # train_nnet, test_nnet = cPickle.load(file(nnet_fname,'rb'))

  # x = T.dmatrix('x')
  # x_q = T.lmatrix('q')
  # x_q_overlap = T.lmatrix('q_overlap')
  # x_a = T.lmatrix('a')
  # x_a_overlap = T.lmatrix('a_overlap')
  # y = T.ivector('y')

  # nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier = test_nnet.layers
  # nnet_q.set_input((x_q, x_q_overlap))
  # nnet_a.set_input((x_a, x_a_overlap))
  # pairwise_layer.set_input((nnet_q.output, nnet_a.output))
  # hidden_layer.set_input(pairwise_layer.output)
  # classifier.set_input(hidden_layer.output)

  # predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

  # batch_x_q = T.lmatrix('batch_x_q')
  # batch_x_a = T.lmatrix('batch_x_a')
  # batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
  # batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
  # batch_y = T.ivector('batch_y')

  # inputs_pred = [batch_x_q,
  #                batch_x_a,
  #                batch_x_q_overlap,
  #                batch_x_a_overlap,
  #                ]

  # givens_pred = {x_q: batch_x_q,
  #                x_a: batch_x_a,
  #                x_q_overlap: batch_x_q_overlap,
  #                x_a_overlap: batch_x_a_overlap,
  #                }

  # pred_prob_fn = theano.function(inputs=inputs_pred,
  #                           outputs=predictions_prob,
  #                           givens=givens_pred)

  ## plot graphs
  # print theano.printing.pprint(predictions_prob)
  # print theano.printing.debugprint(predictions_prob)
  # theano.printing.pydotprint(predictions_prob, outfile="pics/severyn_model_pred_prob.png", var_with_name_simple=True)  