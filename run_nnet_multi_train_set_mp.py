'''
  Train CNN on multiple training sets in parallel. Evalute test examples using all models and vote for the predictions.
  Model saving and loading refers to http://stackoverflow.com/questions/31921084/how-to-save-serialize-a-trained-model-in-theano
  

  Created on August 11, 2016
  Updated on August 18, 2016
  @author: Wei Wei

'''
from datetime import datetime
from sklearn import metrics
from theano import tensor as T
# import subprocess
# import pandas as pd
from tqdm import tqdm
import nn_layers
import sgd_trainer
import os, cPickle, time, math, sys, numpy, theano
from collections import Counter, defaultdict
import multiprocessing as mp 
import warnings
warnings.filterwarnings("ignore")  # TODO remove

print "All modules loaded"

### THEANO DEBUG FLAGS
# theano.config.optimizer = 'fast_compile'
# theano.config.exception_verbosity = 'high'



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

class Consumer(mp.Process):
    def __init__(self,task_queue, result_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        # self.result_queue = result_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" %mp.current_process()
                self.task_queue.task_done()
                break
            answer = next_task.__call__()
            self.task_queue.task_done()
            # self.result_queue.put(answer)
        return
    
class Train_subset(object):
    def __init__(self,model_file, base_dir, sub_dir, out_dir):
        self.model_file = model_file
        self.base_dir = base_dir
        self.data_dir = sub_dir
        self.out_dir = out_dir

    def __call__(self):
        # '''Keep letter-digit combinations, and terms'''
        # tokens = list(set(self.text[:-1].split(' ')))
        # return tokens
        ZEROUT_DUMMY_WORD = True
        numpy_rng = numpy.random.RandomState(123)
        ## Load data
        q_train = numpy.load(os.path.join(self.data_dir, 'train.questions.npy'))
        a_train = numpy.load(os.path.join(self.data_dir, 'train.answers.npy'))
        q_overlap_train = numpy.load(os.path.join(self.data_dir, 'train.q_overlap_indices.npy'))
        a_overlap_train = numpy.load(os.path.join(self.data_dir, 'train.a_overlap_indices.npy'))
        q_knn_count_train = numpy.load(os.path.join(self.data_dir, 'train.q_knn_counts.npy'))
        a_knn_count_train = numpy.load(os.path.join(self.data_dir, 'train.a_knn_counts.npy'))
        q_mti_train = numpy.load(os.path.join(self.data_dir, 'train.q_mti.npy'))
        a_mti_train = numpy.load(os.path.join(self.data_dir, 'train.a_mti.npy'))
        y_train = numpy.load(os.path.join(self.data_dir, 'train.labels.npy'))

        train_nnet, _ = cPickle.load(file(self.model_file, 'rb'))

        params = train_nnet.params

        total_params = sum([numpy.prod(param.shape.eval()) for param in params])
        print 'Total params number:', total_params

        x = T.dmatrix('x')
        x_q = T.lmatrix('q')
        x_q_overlap = T.lmatrix('q_overlap')
        x_q_knn_count = T.lmatrix('q_knn_count')
        x_q_mti = T.lmatrix('q_mti')
        x_a = T.lmatrix('a')
        x_a_overlap = T.lmatrix('a_overlap')
        x_a_knn_count = T.lmatrix('a_knn_count')
        x_a_mti = T.lmatrix('a_mti')
        y = T.ivector('y')

        nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier = train_nnet.layers
        nnet_q.set_input((x_q, x_q_overlap, x_q_knn_count, x_q_mti))
        nnet_a.set_input((x_a, x_a_overlap, x_a_knn_count, x_a_mti))
        pairwise_layer.set_input((nnet_q.output, nnet_a.output))
        hidden_layer.set_input(pairwise_layer.output)
        classifier.set_input(hidden_layer.output)

        test_nnet = train_nnet

        cost = train_nnet.layers[-1].training_cost(y)
        predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

        batch_x_q = T.lmatrix('batch_x_q')
        batch_x_a = T.lmatrix('batch_x_a')
        batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
        batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
        batch_x_q_knn_count = T.lmatrix('batch_x_q_knn_count')
        batch_x_a_knn_count = T.lmatrix('batch_x_a_knn_count')
        batch_x_q_mti = T.lmatrix('batch_x_q_mti')
        batch_x_a_mti = T.lmatrix('batch_x_a_mti')
        batch_y = T.ivector('batch_y')

        updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')

        inputs_pred = [batch_x_q,
                       batch_x_a,
                       batch_x_q_overlap,
                       batch_x_a_overlap,
                       batch_x_q_knn_count,
                       batch_x_a_knn_count,
                       batch_x_q_mti,
                       batch_x_a_mti,
                       ]

        givens_pred = {x_q: batch_x_q,
                       x_a: batch_x_a,
                       x_q_overlap: batch_x_q_overlap,
                       x_a_overlap: batch_x_a_overlap,
                       x_q_knn_count: batch_x_q_knn_count,
                       x_a_knn_count: batch_x_a_knn_count,
                       x_q_mti: batch_x_q_mti,
                       x_a_mti: batch_x_a_mti,                 
                     }

        inputs_train = [batch_x_q,
                     batch_x_a,
                     batch_x_q_overlap,
                     batch_x_a_overlap,
                     batch_x_q_knn_count,
                     batch_x_a_knn_count,
                     batch_x_q_mti,
                     batch_x_a_mti,
                     batch_y,
                     ]

        givens_train = {x_q: batch_x_q,
                     x_a: batch_x_a,
                     x_q_overlap: batch_x_q_overlap,
                     x_a_overlap: batch_x_a_overlap,
                     x_q_knn_count: batch_x_q_knn_count,
                     x_a_knn_count: batch_x_a_knn_count,
                     x_q_mti: batch_x_q_mti,
                     x_a_mti: batch_x_a_mti,   
                     y: batch_y}

        train_fn = theano.function(inputs=inputs_train,
                                 outputs=cost,
                                 updates=updates,
                                 givens=givens_train)

        pred_prob_fn = theano.function(inputs=inputs_pred,
                                outputs=predictions_prob,
                                givens=givens_pred)

        def predict_prob_batch(batch_iterator):
            preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
                                               batch_x_q_knn_count, batch_x_a_knn_count, batch_x_q_mti, batch_x_a_mti) \
                                               for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
                                                   batch_x_q_knn_count, batch_x_a_knn_count, \
                                                   batch_x_q_mti, batch_x_a_mti, _ in batch_iterator])
            return preds[:batch_iterator.n_samples]        

        # train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, q_knn_count_train, a_knn_count_train, q_mti_train, a_mti_train, y_train], batch_size=batch_size, randomize=True)
        # dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, q_knn_count_dev, a_knn_count_dev, q_mti_dev, a_mti_dev, y_dev], batch_size=batch_size, randomize=False)
        # test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
        
        ## For test
        train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, q_knn_count_dev, a_knn_count_dev, q_mti_dev, a_mti_dev, y_dev], batch_size=batch_size, randomize=False)
        dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
        test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
        ## End of test

        labels = sorted(numpy.unique(y_test))
        print 'labels', labels

        print "Zero out dummy word:", ZEROUT_DUMMY_WORD
        if ZEROUT_DUMMY_WORD:
            W_emb_list = [w for w in params if w.name == 'W_emb']
            zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

        best_dev_acc = -numpy.inf
        epoch = 0
        timer_train = time.time()
        no_best_dev_update = 0
        num_train_batches = len(train_set_iterator)

        while epoch < n_epochs:
            timer = time.time()
            for i, (x_q, x_a, x_q_overlap, x_a_overlap, x_q_knn_count, x_a_knn_count, x_q_mti, x_a_mti, y) in enumerate(tqdm(train_set_iterator), 1):
                train_fn(x_q, x_a, x_q_overlap, x_a_overlap, x_q_knn_count, x_a_knn_count, x_q_mti, x_a_mti, y)
                # Make sure the null word in the word embeddings always remains zero

                if ZEROUT_DUMMY_WORD:
                    zerout_dummy_word()

                # if i % 10 == 0 or i == num_train_batches:
                if i == num_train_batches:
                # if i % 100 == 0 or i == num_train_batches:
                    y_pred_dev = predict_prob_batch(dev_set_iterator)
                    dev_acc = metrics.roc_auc_score(y_dev, y_pred_dev) * 100
                    if dev_acc > best_dev_acc:
                        y_pred = predict_prob_batch(test_set_iterator)
                        test_acc = map_score(qids_test, y_test, y_pred) * 100
                        # print('epoch: {} batch: {} dev auc: {:.4f}; test map: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, test_acc, best_dev_acc))
                        best_dev_acc = dev_acc
                        best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
                        no_best_dev_update = 0

            if no_best_dev_update >= 3:
                print "Quitting after of no update of the best score on dev set", no_best_dev_update
                break

            print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
            epoch += 1
            no_best_dev_update += 1

        print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
        for i, param in enumerate(best_params):
            params[i].set_value(param, borrow=True)

        y_pred_test = predict_prob_batch(test_set_iterator)
        test_acc = map_score(qids_test, y_test, y_pred_test) * 100
        print "MAP on test set: ", test_acc/100.0
        print    
        fname = os.path.join(self.out_dir, 'best_dev_params.epoch={:02d};batch={:05d};dev_acc={:.2f}.dat'.format(epoch, i, best_dev_acc))
        numpy.savetxt(os.path.join(self.out_dir, 'test.epoch={:02d};batch={:05d};dev_acc={:.2f}.predictions.npy'.format(epoch, i, best_dev_acc)), y_pred)
        cPickle.dump(best_params, open(fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        fout = file(self.out_dir,"map.txt")
        fout.write(str(test_acc),"w")

            
def train_in_parallel(model_file, base_dir, inDir, outDir, batch_num):
    '''model_file: pickled model file, inDir: training data setset dir, outFile: pickled best parameter file from every training subset'''

    tasks = mp.JoinableQueue()
    results = mp.Queue()

    # num_consumers = mp.cpu_count()
    num_consumers = batch_num
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks, results) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()

    train_data_dirs = os.listdir(inDir)
    for sub_dir_name in train_data_dirs:
        sub_dir = os.path.join(inDir, sub_dir_name)
        sub_out_dir = os.path.join(outDir, sub_dir_name)
        tasks.put(Train_subset(model_file, base_dir, sub_dir, sub_out_dir))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

    # res_ct = 0
    # res_list = []

    # while doc_ct:
    #     df = results.get()
    #     res_list+= df
    #     doc_ct-=1
    #     res_ct+=1

    # doc_freq = Counter(res_list) 
    # cPickle.dump(doc_freq, file(outFile,"w"))
    return  

def main():
    exp_num = sys.argv[1]
    emb_file_name = sys.argv[2]

    ZEROUT_DUMMY_WORD = True     # ZEROUT_DUMMY_WORD = False

    ## Load data
    base_dir = "/home/w2wei/projects/pointwiseLTR/data/knn_sample"
    data_dir = os.path.join(base_dir, exp_num)

    q_train = numpy.load(os.path.join(data_dir, 'train.questions.npy'))
    a_train = numpy.load(os.path.join(data_dir, 'train.answers.npy'))
    q_overlap_train = numpy.load(os.path.join(data_dir, 'train.q_overlap_indices.npy'))
    a_overlap_train = numpy.load(os.path.join(data_dir, 'train.a_overlap_indices.npy'))
    q_knn_count_train = numpy.load(os.path.join(data_dir, 'train.q_knn_counts.npy'))
    a_knn_count_train = numpy.load(os.path.join(data_dir, 'train.a_knn_counts.npy'))
    q_mti_train = numpy.load(os.path.join(data_dir, 'train.q_mti.npy'))
    a_mti_train = numpy.load(os.path.join(data_dir, 'train.a_mti.npy'))
    y_train = numpy.load(os.path.join(data_dir, 'train.labels.npy'))

    global q_dev, a_dev, q_overlap_dev, a_overlap_dev, q_knn_count_dev, a_knn_count_dev, q_mti_dev, a_mti_dev, y_dev, qids_dev
    q_dev = numpy.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = numpy.load(os.path.join(data_dir, 'dev.answers.npy'))
    q_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.q_overlap_indices.npy'))
    a_overlap_dev = numpy.load(os.path.join(data_dir, 'dev.a_overlap_indices.npy'))
    q_knn_count_dev = numpy.load(os.path.join(data_dir, 'dev.q_knn_counts.npy'))
    a_knn_count_dev = numpy.load(os.path.join(data_dir, 'dev.a_knn_counts.npy'))
    q_mti_dev = numpy.load(os.path.join(data_dir, 'dev.q_mti.npy'))
    a_mti_dev = numpy.load(os.path.join(data_dir, 'dev.a_mti.npy'))    
    y_dev = numpy.load(os.path.join(data_dir, 'dev.labels.npy'))
    qids_dev = numpy.load(os.path.join(data_dir, 'dev.qids.npy'))

    global q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test, qids_test
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

    ##### Model definition ####
    numpy_rng = numpy.random.RandomState(123)
    q_max_sent_size = q_train.shape[1]
    a_max_sent_size = a_train.shape[1]

    ## for overlap indicator features
    ndim = 5
    print "Generating random vocabulary for word overlap indicator features with dim:", ndim
    dummy_word_id = numpy.max(a_overlap_train)
    vocab_emb_overlap = numpy_rng.randn(dummy_word_id+1, ndim) * 0.25 ## Gaussian
    vocab_emb_overlap[-1] = 0 ## dummy indicator variable set to zero
    print "Word overlap indicator matrix size: ", vocab_emb_overlap.shape
    print vocab_emb_overlap
    print

    ## for knn count features
    ndim = 5
    print "Generating random vocabulary for knn count features with dim:", ndim
    max_knn_count = 20 # restricted by 20 nearest neighbors
    vocab_emb_knn_count = numpy_rng.randn(max_knn_count+1, ndim) # one additional row for 0
    vocab_emb_knn_count[0] = 0 ## the first row, idx=0, is for invalid examples
    print "KNN count matrix size: ", vocab_emb_knn_count.shape
    print

    ## for mti features
    ndim = 5
    print "Generating random vocabulary for mti prediction features with dim:", ndim
    max_mti_pred = 2
    vocab_emb_mti = numpy_rng.randn(max_mti_pred+1, ndim)
    vocab_emb_mti[0] = 0
    print "MTI pred matrix size: ", vocab_emb_mti.shape
    print

    ## for words in sentences
    # Load word2vec embeddings
    util_dir = "/home/w2wei/projects/pointwiseLTR/data/utils"
    emb_file_name = sys.argv[2]
    fname = os.path.join(util_dir, emb_file_name) ## 10K training data
    print "Loading word embeddings from ", emb_file_name
    vocab_emb = numpy.load(fname)
    ndim = vocab_emb.shape[1]
    dummpy_word_idx = numpy.max(a_train)
    print "Word embedding matrix size:", vocab_emb.shape

    ## Define model variables
    x = T.dmatrix('x')
    x_q = T.lmatrix('q')
    x_q_overlap = T.lmatrix('q_overlap')
    x_q_knn_count = T.lmatrix('q_knn_count')
    x_q_mti = T.lmatrix('q_mti')
    x_a = T.lmatrix('a')
    x_a_overlap = T.lmatrix('a_overlap')
    x_a_knn_count = T.lmatrix('a_knn_count')
    x_a_mti = T.lmatrix('a_mti')
    y = T.ivector('y')

    #######
    global max_norm, n_outs, n_epochs, batch_size, learning_rate
    n_outs = 2
    n_epochs = 5
    batch_size = 5
    learning_rate = 0.1
    max_norm = 0

    ## 1st conv layer.
    '''adjust ndim and include the knn features'''
    ndim = vocab_emb.shape[1] + vocab_emb_overlap.shape[1] + vocab_emb_knn_count.shape[1] + vocab_emb_mti.shape[1]

    ### Nonlinearity type
    # activation = nn_layers.relu_f
    activation = T.tanh

    # dropout_rate = 0.5
    nkernels = 100 # filter number
    q_k_max = 1 # k max pooling
    a_k_max = 1 # k max pooling

    # filter_widths = [3,4,5]
    q_filter_widths = [5]
    a_filter_widths = [5]

    ###### QUESTION ######
    ## return padded tensor as part of input for the conv layer
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
    # print "lookup_table_words weights: ", len(lookup_table_words.weights), lookup_table_words.W.get_value().shape
    # note: layer lookup_table_words, self.W is not added to list self.weights

    lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)
    # print "lookup_table_overlap weights: ", len(lookup_table_overlap.weights), lookup_table_overlap.W.get_value().shape
    # note: layer lookup_table_overlap, self.W is added to list self.weights
    
    lookup_table_knn_count = nn_layers.LookupTableFast(W=vocab_emb_knn_count, pad=max(q_filter_widths)-1)
    ## vocab_emb_knn_count shape 21*5

    lookup_table_mti = nn_layers.LookupTableFast(W=vocab_emb_mti, pad=max(q_filter_widths)-1)
    ## vocab_emb_mti shape 3*5

    # lookup_table returns a 4D tensor
    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap, lookup_table_knn_count, lookup_table_mti])
    # note: lookup_table, self.weights only contains W from lookup_table_overlap, but not lookup_table_words

    ## conv layer input data shape, notice this is the input shape after padding. 
    num_input_channels = 1
    input_shape = (batch_size, num_input_channels, q_max_sent_size + 2*(max(q_filter_widths)-1), ndim)
    ## input batch is a 4D tensor. 1: batch size, 
                         #  2: nkernels, 
                         #  3: input_shape[2]-filter_shape+1 (in valid mode) = 495,
                         #  4: ndim = 110. 100 for word emb, 5 for overlap indicator feat, 5 for knn count feat
    conv_layers = []
    for filter_width in q_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
        # print "conv: ", type(conv), conv.get_value().shape
        # raw_input("check conv shape...")
        ## conv is a 4D tensor. 1: batch size, 
                             #  2: nkernels, 
                             #  3: input_shape[2]-filter_shape+1 (in valid mode) = 495,
                             #  4: conv value, scalar
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
        # print 'non_linearity: ', non_linearity.type, non_linearity.get_value().shape
        # raw_input("check non_linearity shape...")
        ## non_linearity is a 4D tensor. 1: batch size, 
                             #  2: nkernels, 
                             #  3: input_shape[2]-filter_shape+1 (in valid mode) = 495,
                             #  4: tanh(conv value), scalar
        pooling = nn_layers.KMaxPoolLayer(k_max=q_k_max) # return T.max(input, axis=2)
        # print "pooling: ", pooling.type, pooling.get_value().shape
        # raw_input("check pooling shape...")
        ## pooling is a 3D tensor. 1: batch size, 
                             #  2: nkernels, 
                             #  3: max tanh(conv value), scalar
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
        ## conv2dNonLinearMaxPool is a 3D tensor.1: batch size, 
                             #  2: nkernels, 
                             #  3: max tanh(conv value), scalar
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    ## join_layer is a 3D tensor. 1: batch size, 
                               #  2: nkernels, 
                               #  3: max tanh(conv value) vector
    flatten_layer = nn_layers.FlattenLayer() ## flatten in axis 2
    ## flatten_layer is a 2D maxtrix. 1. batch size,
                                #    2. nkernels, value: a list of max values from all filters
    nnet_q = nn_layers.FeedForwardNet(layers=[
                                  lookup_table,
                                  join_layer,
                                  flatten_layer,
                                  ])
    nnet_q.set_input((x_q, x_q_overlap, x_q_knn_count, x_q_mti))

    ###### ANSWER ######
    lookup_table_words = nn_layers.LookupTableFastStatic(W=vocab_emb, pad=max(q_filter_widths)-1)
    lookup_table_overlap = nn_layers.LookupTableFast(W=vocab_emb_overlap, pad=max(q_filter_widths)-1)
    lookup_table_knn_count = nn_layers.LookupTableFast(W=vocab_emb_knn_count, pad=max(q_filter_widths)-1)
    lookup_table_mti = nn_layers.LookupTableFast(W=vocab_emb_mti, pad=max(q_filter_widths)-1)
    lookup_table = nn_layers.ParallelLookupTable(layers=[lookup_table_words, lookup_table_overlap, lookup_table_knn_count, lookup_table_mti])

    # num_input_channels = len(lookup_table.layers)
    input_shape = (batch_size, num_input_channels, a_max_sent_size + 2*(max(a_filter_widths)-1), ndim)
    conv_layers = []
    for filter_width in a_filter_widths:
        filter_shape = (nkernels, num_input_channels, filter_width, ndim)
        conv = nn_layers.Conv2dLayer(rng=numpy_rng, filter_shape=filter_shape, input_shape=input_shape)
        non_linearity = nn_layers.NonLinearityLayer(b_size=filter_shape[0], activation=activation)
        pooling = nn_layers.KMaxPoolLayer(k_max=a_k_max)
        conv2dNonLinearMaxPool = nn_layers.FeedForwardNet(layers=[conv, non_linearity, pooling])
        conv_layers.append(conv2dNonLinearMaxPool)

    join_layer = nn_layers.ParallelLayer(layers=conv_layers)
    flatten_layer = nn_layers.FlattenLayer()

    nnet_a = nn_layers.FeedForwardNet(layers=[
                                  lookup_table,
                                  join_layer,
                                  flatten_layer,
                                  ])
    nnet_a.set_input((x_a, x_a_overlap, x_a_knn_count, x_a_mti))

    q_logistic_n_in = nkernels * len(q_filter_widths) * q_k_max
    print "q_logistic_n_in: ", q_logistic_n_in
    a_logistic_n_in = nkernels * len(a_filter_widths) * a_k_max
    print "a_logistic_n_in: ", a_logistic_n_in
    print 

    pairwise_layer = nn_layers.PairwiseNoFeatsLayer(q_in=q_logistic_n_in, a_in=a_logistic_n_in)
    pairwise_layer.set_input((nnet_q.output, nnet_a.output))

    n_in = q_logistic_n_in + a_logistic_n_in + 1

    hidden_layer = nn_layers.LinearLayer(numpy_rng, n_in=n_in, n_out=n_in, activation=activation)
    hidden_layer.set_input(pairwise_layer.output)

    classifier = nn_layers.LogisticRegression(n_in=n_in, n_out=n_outs)
    classifier.set_input(hidden_layer.output)


    train_nnet = nn_layers.FeedForwardNet(layers=[nnet_q, nnet_a, pairwise_layer, hidden_layer, classifier], name="Training nnet")
    test_nnet = train_nnet

    print train_nnet

    params = train_nnet.params

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    exp_num = sys.argv[1]
    nnet_outdir = 'exp.out/{}; ndim={};batch={};max_norm={};learning_rate={};{}'.format(exp_num, ndim, batch_size, max_norm, learning_rate, ts)
    if not os.path.exists(nnet_outdir):
        os.makedirs(nnet_outdir)
    nnet_fname = os.path.join(nnet_outdir, 'nnet.dat')
    print "Saving to", nnet_fname
    cPickle.dump([train_nnet, test_nnet], open(nnet_fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    total_params = sum([numpy.prod(param.shape.eval()) for param in params])
    print 'Total params number:', total_params

    subset_dir = os.path.join(data_dir, "train")
    subres_dir = os.path.join(data_dir, "train_params")
    if not os.path.exists(subres_dir):
        os.makedirs(subres_dir)

    train_in_parallel(nnet_fname, data_dir, subset_dir, subres_dir, batch_num=10)


    # ### 
    # cost = train_nnet.layers[-1].training_cost(y)

    # predictions = test_nnet.layers[-1].y_pred
    # predictions_prob = test_nnet.layers[-1].p_y_given_x[:,-1]

    # batch_x_q = T.lmatrix('batch_x_q')
    # batch_x_a = T.lmatrix('batch_x_a')
    # batch_x_q_overlap = T.lmatrix('batch_x_q_overlap')
    # batch_x_a_overlap = T.lmatrix('batch_x_a_overlap')
    # batch_x_q_knn_count = T.lmatrix('batch_x_q_knn_count')
    # batch_x_a_knn_count = T.lmatrix('batch_x_a_knn_count')
    # batch_x_q_mti = T.lmatrix('batch_x_q_mti')
    # batch_x_a_mti = T.lmatrix('batch_x_a_mti')
    # batch_y = T.ivector('batch_y')

    # updates = sgd_trainer.get_adadelta_updates(cost, params, rho=0.95, eps=1e-6, max_norm=max_norm, word_vec_name='W_emb')

    # inputs_pred = [batch_x_q,
    #                batch_x_a,
    #                batch_x_q_overlap,
    #                batch_x_a_overlap,
    #                batch_x_q_knn_count,
    #                batch_x_a_knn_count,
    #                batch_x_q_mti,
    #                batch_x_a_mti,
    #                ]

    # givens_pred = {x_q: batch_x_q,
    #                x_a: batch_x_a,
    #                x_q_overlap: batch_x_q_overlap,
    #                x_a_overlap: batch_x_a_overlap,
    #                x_q_knn_count: batch_x_q_knn_count,
    #                x_a_knn_count: batch_x_a_knn_count,
    #                x_q_mti: batch_x_q_mti,
    #                x_a_mti: batch_x_a_mti,                 
    #              }

    # inputs_train = [batch_x_q,
    #              batch_x_a,
    #              batch_x_q_overlap,
    #              batch_x_a_overlap,
    #              batch_x_q_knn_count,
    #              batch_x_a_knn_count,
    #              batch_x_q_mti,
    #              batch_x_a_mti,
    #              batch_y,
    #              ]

    # givens_train = {x_q: batch_x_q,
    #              x_a: batch_x_a,
    #              x_q_overlap: batch_x_q_overlap,
    #              x_a_overlap: batch_x_a_overlap,
    #              x_q_knn_count: batch_x_q_knn_count,
    #              x_a_knn_count: batch_x_a_knn_count,
    #              x_q_mti: batch_x_q_mti,
    #              x_a_mti: batch_x_a_mti,   
    #              y: batch_y}

    # train_fn = theano.function(inputs=inputs_train,
    #                          outputs=cost,
    #                          updates=updates,
    #                          givens=givens_train)

    # pred_prob_fn = theano.function(inputs=inputs_pred,
    #                         outputs=predictions_prob,
    #                         givens=givens_pred)

    # def predict_prob_batch(batch_iterator):
    #     preds = numpy.hstack([pred_prob_fn(batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
    #                                        batch_x_q_knn_count, batch_x_a_knn_count, batch_x_q_mti, batch_x_a_mti) \
    #                                        for batch_x_q, batch_x_a, batch_x_q_overlap, batch_x_a_overlap, \
    #                                            batch_x_q_knn_count, batch_x_a_knn_count, \
    #                                            batch_x_q_mti, batch_x_a_mti, _ in batch_iterator])
    #     return preds[:batch_iterator.n_samples]        

    # # train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_train, a_train, q_overlap_train, a_overlap_train, q_knn_count_train, a_knn_count_train, q_mti_train, a_mti_train, y_train], batch_size=batch_size, randomize=True)
    # # dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, q_knn_count_dev, a_knn_count_dev, q_mti_dev, a_mti_dev, y_dev], batch_size=batch_size, randomize=False)
    # # test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
    
    # ## For test
    # train_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_dev, a_dev, q_overlap_dev, a_overlap_dev, q_knn_count_dev, a_knn_count_dev, q_mti_dev, a_mti_dev, y_dev], batch_size=batch_size, randomize=False)
    # dev_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
    # test_set_iterator = sgd_trainer.MiniBatchIteratorConstantBatchSize(numpy_rng, [q_test, a_test, q_overlap_test, a_overlap_test, q_knn_count_test, a_knn_count_test, q_mti_test, a_mti_test, y_test], batch_size=batch_size, randomize=False)
    # ## End of test

    # labels = sorted(numpy.unique(y_test))
    # print 'labels', labels

    # print "Zero out dummy word:", ZEROUT_DUMMY_WORD
    # if ZEROUT_DUMMY_WORD:
    #     W_emb_list = [w for w in params if w.name == 'W_emb']
    #     zerout_dummy_word = theano.function([], updates=[(W, T.set_subtensor(W[-1:], 0.)) for W in W_emb_list])

    # best_dev_acc = -numpy.inf
    # epoch = 0
    # timer_train = time.time()
    # no_best_dev_update = 0
    # num_train_batches = len(train_set_iterator)

    # while epoch < n_epochs:
    #     timer = time.time()
    #     for i, (x_q, x_a, x_q_overlap, x_a_overlap, x_q_knn_count, x_a_knn_count, x_q_mti, x_a_mti, y) in enumerate(tqdm(train_set_iterator), 1):
    #         train_fn(x_q, x_a, x_q_overlap, x_a_overlap, x_q_knn_count, x_a_knn_count, x_q_mti, x_a_mti, y)
    #         # Make sure the null word in the word embeddings always remains zero

    #         if ZEROUT_DUMMY_WORD:
    #             zerout_dummy_word()

    #         # if i % 10 == 0 or i == num_train_batches:
    #         if i == num_train_batches:
    #         # if i % 100 == 0 or i == num_train_batches:
    #             y_pred_dev = predict_prob_batch(dev_set_iterator)
    #             dev_acc = metrics.roc_auc_score(y_dev, y_pred_dev) * 100
    #             if dev_acc > best_dev_acc:
    #                 y_pred = predict_prob_batch(test_set_iterator)
    #                 test_acc = map_score(qids_test, y_test, y_pred) * 100

    #                 print('epoch: {} batch: {} dev auc: {:.4f}; test map: {:.4f}; best_dev_acc: {:.4f}'.format(epoch, i, dev_acc, test_acc, best_dev_acc))
    #                 best_dev_acc = dev_acc
    #                 best_params = [numpy.copy(p.get_value(borrow=True)) for p in params]
    #                 no_best_dev_update = 0

    #     if no_best_dev_update >= 3:
    #         print "Quitting after of no update of the best score on dev set", no_best_dev_update
    #         break

    #     print('epoch {} took {:.4f} seconds'.format(epoch, time.time() - timer))
    #     epoch += 1
    #     no_best_dev_update += 1

    # print('Training took: {:.4f} seconds'.format(time.time() - timer_train))
    # for i, param in enumerate(best_params):
    #     params[i].set_value(param, borrow=True)

    # y_pred_test = predict_prob_batch(test_set_iterator)
    # test_acc = map_score(qids_test, y_test, y_pred_test) * 100
    # print "MAP on test set: ", test_acc/100.0
    # print    
    # # fname = os.path.join(nnet_outdir, 'best_dev_params.epoch={:02d};batch={:05d};dev_acc={:.2f}.dat'.format(epoch, i, best_dev_acc))
    # # numpy.savetxt(os.path.join(nnet_outdir, 'test.epoch={:02d};batch={:05d};dev_acc={:.2f}.predictions.npy'.format(epoch, i, best_dev_acc)), y_pred)
    # # cPickle.dump(best_params, open(fname, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
