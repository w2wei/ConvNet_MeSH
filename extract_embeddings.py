import numpy as np
import cPickle
import os, sys
from gensim import models
from alphabet import Alphabet
# from utils import load_bin_vec


# def load_senna_vec():
#     # word2vec = {}
#     words = np.loadtxt('../data/words.lst', dtype='str')
#     vecs = np.loadtxt('../data/embeddings.txt')
#     word2vec = dict(zip(words, vecs))
#     return word2vec

def load_bin_vec(fmodel, train_vocab):
    '''Load trained word2vec vectors from PMCOA'''
    model = models.Word2Vec.load(fmodel)
    model_vocab = model.vocab.keys()
    train_vocab = set(train_vocab)
    word_vecs = {}
    for term in model_vocab:
        if term in train_vocab:
            word_vecs[term]=model[term]
    print "model vocab size: ", len(model_vocab)
    return word_vecs

def main():
    np.random.seed(123)
    base_dir = "/home/w2wei/projects/pointwiseLTR/data/knn_sample"
    data_dir = os.path.join(base_dir, sys.argv[1])
    # data_dir = "/home/w2wei/projects/pointwiseLTR/data/sample/SAMPLE_TRAIN"
    model_dir = "/home/w2wei/projects/word2vec/models/"
    # model_file = os.path.join(model_dir, "dim100_sample_10K_win5_model.ml") ## 10K training data

    model_file = os.path.join(model_dir, "dim100_sample_1M_window_5.ml") ## 1.3M training data
    # model_file = os.path.join(model_dir, "dim100_sample_10K_win5_model.ml") ## 1.3M training data
    # util_dir = "/home/w2wei/projects/pointwiseLTR/data/utils"

    fname_vocab = os.path.join(data_dir, 'vocab.pickle')

    alphabet = cPickle.load(open(fname_vocab))
    train_vocab = alphabet.keys()
    print "training set vocab size", len(alphabet)

    word2vec = load_bin_vec(model_file, train_vocab) # a dictionary
    print "word2vec size: ", len(word2vec)
    ndim = len(word2vec[word2vec.keys()[0]])
    print 'ndim', ndim

    random_words_count = 0
    vocab_emb = np.zeros((len(alphabet) + 1, ndim))

    # for word, idx in alphabet.iteritems():
    #     # word_vec = word2vec.get(word, None)
    #     word_vec = np.random.uniform(-0.25, 0.25, ndim)
    #     random_words_count += 1
    #     vocab_emb[idx] = word_vec
    for word, idx in alphabet.iteritems():
        word_vec = word2vec.get(word, None)
        if word_vec is None:
            word_vec = np.random.uniform(-0.25, 0.25, ndim)
            random_words_count += 1
        vocab_emb[idx] = word_vec
    print "Using zero vector as random"
    print 'random_words_count', random_words_count
    print vocab_emb.shape
    # outfile = os.path.join(data_dir, 'emb_all_random_dim50.npy')
    outfile = os.path.join(data_dir, 'emb_{}.npy'.format(os.path.basename(model_file)))
    print "emb file: ", outfile
    np.save(outfile, vocab_emb)

if __name__ == '__main__':
  main()
