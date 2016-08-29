'''
    Find KNN documents for given PMIDs from documents in D2V+TFIDF representations
    
    Created on July 24, 2016
    Updated on August 8, 2016
    @author: Wei Wei
'''

import sys, gensim, os, collections, random, time, re, string, cPickle, sys
import numpy as np
from scipy import spatial
from numpy import linalg as LA
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import multiprocessing as mp
from collections import defaultdict, Counter
from operator import itemgetter

global train_set_d2v_dict
global train_set_tfidf_dict
import corpus_medline_tfidf_for_MEDLINE95_97 as tfidf9597

class Consumer(mp.Process):
    def __init__(self,task_queue, result_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" %proc_name#mp.current_process()
                self.task_queue.task_done()
                break
            answer = next_task()
            # answer = next_task.__call__()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class CleanTestCorpusTask(object):
    def __init__(self, pmid, text):
        self.pmid = pmid
        self.text = text
    
    def __call__(self):
        '''Keep letter-digit combinations, and terms'''
        if len(self.text)==3:
            text = self.text[0]+self.text[1]
            text = " ".join(text)
            doc = []
            sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") # a sentence list from doc 
            if sent_tokenize_list: # if sent_tokenize_list is not empty
                for sent in sent_tokenize_list:
                    words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                    words = [word.strip(string.punctuation) for word in words]
                    words = [word for word in words if not word in stopwords.words("english")]
                    words = [word for word in words if len(word)>1] # remove single letters and non alphabetic characters
                    words = [word for word in words if re.search('[a-zA-Z]',word)] # remove digits
                    doc+=words
            doc = filter(None, doc)
            return [self.pmid, doc]
        # except Exception as e:
        #     print "build_test_corpus error, pmid ",self.pmid
        #     print e
        #     raw_input('wait...')
        #     return [self.pmid, '']

class D2V_TFIDF_Task(object):
    '''Represent text using D2V+TFIDF repr'''
    def __init__(self, test_d2v_vec, test_tfidf_dict, test_pmid):
        self.test_d2v_vec = test_d2v_vec
        self.test_tfidf_dict = test_tfidf_dict
        self.test_pmid = test_pmid

    def __call__(self):
        sim_list = [] ## list of similarity tuples, [(train_pmid, cos_sim)]

        for train_pmid, _ in train_set_tfidf_dict.iteritems():
            train_d2v_vec = train_set_d2v_dict.get(train_pmid)
            train_tfidf_dict = train_set_tfidf_dict.get(train_pmid)

            p1 = np.dot(self.test_d2v_vec, train_d2v_vec)

            test_tfidf_dict_term_idx = self.test_tfidf_dict.keys()
            train_tfidf_dict_term_idx = train_tfidf_dict.keys()
            overlap_tfidf_term_idx = list(set(test_tfidf_dict_term_idx)&set(train_tfidf_dict_term_idx))
            train_tfidf_vec = []
            test_tfidf_vec = []
            for term_idx in overlap_tfidf_term_idx:
                train_tfidf_vec.append(train_tfidf_dict.get(term_idx))
                test_tfidf_vec.append(self.test_tfidf_dict.get(term_idx))
            train_tfidf_vec = np.array(train_tfidf_vec)#.reshape(1,len(train_tfidf_vec))
            test_tfidf_vec = np.array(test_tfidf_vec)#.reshape(1,len(test_tfidf_vec))
            p2 = np.dot(test_tfidf_vec, train_tfidf_vec)

            p3 = LA.norm(np.hstack((self.test_d2v_vec, test_tfidf_vec)))
            p4 = LA.norm(np.hstack((train_d2v_vec.T, train_tfidf_vec)))

            cos_sim = (p1+p2)/(p3*p4)
            sim_list.append((train_pmid, cos_sim))
        sorted_sim_list = sorted(sim_list,key=itemgetter(1),reverse=True)
        sorted_sim_list = sorted_sim_list[:100] # return the top 100 similar docs
        return [self.test_pmid, sorted_sim_list]

class D2V_Task(object):
    '''Represent text using D2V'''
    def __init__(self, test_d2v_vec, test_tfidf_dict, test_pmid):
        self.test_d2v_vec = test_d2v_vec
        self.test_tfidf_dict = test_tfidf_dict
        self.test_pmid = test_pmid

    def __call__(self):
        sim_list = [] ## list of similarity tuples, [(train_pmid, cos_sim)]

        for train_pmid, _ in train_set_tfidf_dict.iteritems():
            train_d2v_vec = train_set_d2v_dict.get(train_pmid)
            cos_sim = 1-spatial.distance.cosine(self.test_d2v_vec, train_d2v_vec)
            sim_list.append((train_pmid, cos_sim))
        sorted_sim_list = sorted(sim_list,key=itemgetter(1),reverse=True)
        sorted_sim_list = sorted_sim_list[:100] # return the top 100 similar docs
        return [self.test_pmid, sorted_sim_list]

class TFIDF_Task(object):
    '''Represent text using D2V+TFIDF repr'''
    def __init__(self, test_d2v_vec, test_tfidf_dict, test_pmid):
        # self.test_d2v_vec = test_d2v_vec
        self.test_tfidf_dict = test_tfidf_dict
        self.test_pmid = test_pmid

    def __call__(self):
        sim_list = [] ## list of similarity tuples, [(train_pmid, cos_sim)]

        for train_pmid, _ in train_set_tfidf_dict.iteritems():
            train_tfidf_dict = train_set_tfidf_dict.get(train_pmid)

            test_tfidf_dict_term_idx = self.test_tfidf_dict.keys()
            train_tfidf_dict_term_idx = train_tfidf_dict.keys()
            overlap_tfidf_term_idx = list(set(test_tfidf_dict_term_idx)&set(train_tfidf_dict_term_idx))
            train_tfidf_vec = []
            test_tfidf_vec = []
            for term_idx in overlap_tfidf_term_idx:
                train_tfidf_vec.append(train_tfidf_dict.get(term_idx))
                test_tfidf_vec.append(self.test_tfidf_dict.get(term_idx))
            train_tfidf_vec = np.array(train_tfidf_vec)#.reshape(1,len(train_tfidf_vec))
            test_tfidf_vec = np.array(test_tfidf_vec)#.reshape(1,len(test_tfidf_vec))
            try:
                cos_sim = 1-spatial.distance.cosine(test_tfidf_vec, train_tfidf_vec)
                sim_list.append((train_pmid, cos_sim))
            except Exception as e:
                continue
        sorted_sim_list = sorted(sim_list,key=itemgetter(1),reverse=True)
        sorted_sim_list = sorted_sim_list[:100] # return the top 100 similar docs
        return [self.test_pmid, sorted_sim_list]

def load_train_corpus(train_corpus_file):
    '''Load traning set docs'''
    train_corpus = read_corpus(train_corpus_file)
    return train_corpus

def infer_tfidf(idf_file, test_corpus, docid_pmid_dict):
    '''infer tf-idf repr for docs in test corpus'''
    try:
        idf_dict = cPickle.load(file(idf_file)) ## load idf
    except:
        ## generate idf_dict
        period = "1995_1997"
        bioasq_base_dir = "/home/w2wei/data"
        bioasq_gensim_dir = os.path.join(bioasq_base_dir, "gensim")
        bioasq_doc2vec_input_file = os.path.join(bioasq_gensim_dir,"medline_%s_all_docs.txt"%period) # One doc per line
        bioasq_tfidf_dir = os.path.join(bioasq_base_dir, "medline_%s_tfidf"%period)
        if not os.path.exists(bioasq_tfidf_dir):
            os.makedirs(bioasq_tfidf_dir)
        bioasq_df_file = os.path.join(bioasq_tfidf_dir, "medline_%s_doc_freq.pkl"%period)
        bioasq_idf_file= os.path.join(bioasq_tfidf_dir, "medline_%s_idf.pkl"%period)
        
        if not os.path.exists(bioasq_df_file):
            tfidf9597.count_df(bioasq_doc2vec_input_file, bioasq_df_file)
        print "MEDLINE 1995-1997 doc freq ready"

        tfidf9597.build_idf(bioasq_df_file, bioasq_idf_file)
        print "MEDLINE 1995-1997 IDF ready"
        idf_dict = cPickle.load(file(idf_file))
        
    docid_pmid_dict = cPickle.load(file(docid_pmid_dict)) ## load doc id - pmid dict

    vocab = idf_dict.keys() # 1558828
    vocab_size = len(vocab)

    ## build a vocab index    
    vocab_idx_dict = defaultdict()
    vocab_idx = 0
    for vocab_idx in xrange(len(vocab)):
        vocab_idx_dict[vocab[vocab_idx]]=vocab_idx

    inferred_vec_dict = defaultdict()
    
    for doc_id in xrange(len(test_corpus)):
        pmid = docid_pmid_dict.get(doc_id)
        tfidf_dict = defaultdict()
        # token_vec = np.zeros((vocab_size, 1))

        text = test_corpus[doc_id].words
        tf = Counter(text)
        for term, freq in tf.iteritems():
            idf = idf_dict.get(term, 25.0) ## set idf=25 for missing terms
            tfidf_dict[vocab_idx_dict.get(term)]=freq*idf
            # token_vec[vocab_idx_dict.get(term),0]=freq*idf #tfidf_dict[term]
        inferred_vec_dict[pmid] = tfidf_dict
    return inferred_vec_dict

def load_d2v_model(model_file):
    '''Load trained D2V model'''
    model = gensim.models.doc2vec.Doc2Vec.load(model_file)
    return model

def infer_d2v(model, test_corpus, docid_pmid_dict):
    '''Infer vectors for every doc in test set using trained model'''
    inferred_vec_dict=defaultdict()
    doc_pmid_dict = cPickle.load(file(docid_pmid_dict))
    for doc_id in xrange(len(test_corpus)):
        inferred_vector = model.infer_vector(test_corpus[doc_id].words)
        pmid=doc_pmid_dict[doc_id]
        inferred_vec_dict[pmid]=inferred_vector
    return inferred_vec_dict

def load_test_corpus(raw_test_file, clean_test_file, doc_index_file):
    if not (os.path.exists(clean_test_file) and os.path.exists(doc_index_file)):
        ## Load raw data
        corpus = cPickle.load(file(raw_test_file))
        ## clean raw data
        clean_test_corpus(corpus, clean_test_file, doc_index_file)
    test_corpus = read_corpus(clean_test_file)
    return test_corpus

def clean_test_corpus(corpus, outFile, outIdxFile):
    '''Process the intermediate format of Lu's data. Save all clean documents into a file, one doc per line. 
       Save the relation between doc id (i.e., line number) and pmid'''
    tasks = mp.JoinableQueue()
    results = mp.Queue()

    num_consumers = mp.cpu_count()
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks, results) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    job_num = 0
    for pmid, raw_text in corpus.iteritems():
        job_num+=1
        tasks.put(CleanTestCorpusTask(pmid, raw_text))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

    falldocs = file(outFile,"w")
    docid_pmid_idx = defaultdict()
    docid = 0
    while job_num:
        pmid, clean_text = results.get()
        clean_text = " ".join(clean_text)+"\n"
        falldocs.write(clean_text)
        docid_pmid_idx[docid]=pmid
        job_num-=1
        docid+=1
    cPickle.dump(docid_pmid_idx, file(outIdxFile,"w"), protocol=cPickle.HIGHEST_PROTOCOL)
    return

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def load_train_d2v(d2v_model, docid_pmid_dict_file, train_d2v_dict_file):
    '''Load d2v vec of docs in training set'''
    try:
        # print 1.0/0
        t0=time.time()
        train_d2v_dict = cPickle.load(file(train_d2v_dict_file,"rb"))
        t1=time.time()
        print "train_d2v_dict loading time: ", t1-t0
    except Exception as e:
        print "Method: load_train_d2v"
        print e       
        docid_pmid_dict = cPickle.load(file(docid_pmid_dict_file))
        train_d2v_dict = defaultdict()
        doc_num = len(docid_pmid_dict)
        print "load_train_d2v doc num ", doc_num
        for docid in xrange(doc_num):
            pmid = docid_pmid_dict.get(docid)
            vec = d2v_model.docvecs[docid]
            train_d2v_dict[pmid]=vec
        print "load_train_d2v train_d2v_dict ready"
        t0=time.time()
        cPickle.dump(train_d2v_dict, file(train_d2v_dict_file,"wb"), protocol=cPickle.HIGHEST_PROTOCOL) ## 50s
        t1=time.time()
        print "pickle train_d2v_dict time: ", t1-t0
    return train_d2v_dict

def load_train_tfidf(idf_file, train_corpus, docid_pmid_dict_file, train_tfidf_dict_file): # train_vocab_file
    '''Load tf-idf repr for docs in training set, save the Counter dictionary since the sparse array is too large'''
    try:
        # print 1.0/0
        t0=time.time()
        inferred_vec_dict = cPickle.load(file(train_tfidf_dict_file,"rb"))
        t1=time.time()
        print "loading train_tfidf_dict time: ",t1-t0
    except Exception as e:
        print "Method: load_train_tfidf"
        print e 
        t0 = time.time()
        docid_pmid_dict = cPickle.load(file(docid_pmid_dict_file))
        idf_dict = cPickle.load(file(idf_file)) ## load idf
        vocab = idf_dict.keys() # 1558828
        vocab_size = len(vocab)
        
        ## build a vocab index    
        vocab_idx_dict = defaultdict()
        vocab_idx = 0
        for vocab_idx in xrange(len(vocab)):
            vocab_idx_dict[vocab[vocab_idx]]=vocab_idx

        doc_num = len(docid_pmid_dict)

        inferred_vec_dict = defaultdict()
        for doc_id in xrange(doc_num):
            pmid = docid_pmid_dict.get(doc_id)
            tfidf_per_doc_dict = defaultdict() ## tf-idf for every doc, key is the index of token in the vocabulary
            text = next(train_corpus).words
            tf = Counter(text)
            for term, freq in tf.iteritems():
                idf = idf_dict.get(term, 25.0) ## set idf=25 for missing terms
                tfidf_per_doc_dict[vocab_idx_dict.get(term)] = freq*idf
            inferred_vec_dict[pmid] = tfidf_per_doc_dict
        cPickle.dump(inferred_vec_dict, file(train_tfidf_dict_file,"wb"), protocol=cPickle.HIGHEST_PROTOCOL)
        t1 = time.time()
        print "build train tfidf time: ", t1-t0
    return inferred_vec_dict

def find_knn(test_set_d2v_dict, test_set_tfidf_dict, train_set_d2v_dict, train_set_tfidf_dict, test_set_pmid_knn_dict_file, rep='d2v+tfidf'):
    '''Find KNN using d2v+tfidf repr, multiprocessing'''
    t0=time.time()
    try:
        tasks = mp.JoinableQueue()
        results = mp.Queue()

        num_consumers = mp.cpu_count()
        print "creating %d consumers "%num_consumers
        consumers = [Consumer(tasks, results) for i in xrange(num_consumers)]
        
        for w in consumers:
            w.start()
        
        job_num = 0
        # sample_size = 200
        test_pmid_list = test_set_d2v_dict.keys()
        if rep.lower()=='d2v+tfidf':
            for test_pmid in test_pmid_list:#[:sample_size]:
                test_d2v_vec, test_tfidf_dict = [test_set_d2v_dict.get(test_pmid), test_set_tfidf_dict.get(test_pmid)]
                tasks.put(D2V_TFIDF_Task(test_d2v_vec, test_tfidf_dict, test_pmid))
                job_num+=1

        elif rep.lower()=='d2v':
            for test_pmid in test_pmid_list:#[:sample_size]:
                test_d2v_vec, test_tfidf_dict = [test_set_d2v_dict.get(test_pmid), test_set_tfidf_dict.get(test_pmid)]
                tasks.put(D2V_Task(test_d2v_vec, test_tfidf_dict, test_pmid))
                job_num+=1            

        elif rep.lower()=='tfidf':
            for test_pmid in test_pmid_list:#[:sample_size]:
                test_d2v_vec, test_tfidf_dict = [test_set_d2v_dict.get(test_pmid), test_set_tfidf_dict.get(test_pmid)]
                tasks.put(TFIDF_Task(test_d2v_vec, test_tfidf_dict, test_pmid))
                job_num+=1                     
        else:
            sys.exit("Incorrect repr value. Must be from ['d2v+tfidf', 'd2v','tfidf']")

        for i in xrange(num_consumers):
            tasks.put(None)
    finally:
        tasks.join()
    print "tasks joined"

    try:
        test_pmid_knn_dict = {}
        while job_num:
            job_num-=1
            test_pmid, knn_list = results.get()
            test_pmid_knn_dict[test_pmid] = knn_list

        cPickle.dump(test_pmid_knn_dict, file(test_set_pmid_knn_dict_file,"wb"),protocol=cPickle.HIGHEST_PROTOCOL)
        print "test_pmid_knn_dict saved"
    except Exception as e:
        print "Saving error "
        print e
    t1=time.time()
    print 'Find knn time cost: ', t1-t0
    return

if __name__=="__main__":
    ## parameter
    period = sys.argv[1]
    try:
        startyear, endyear = period.split("-")
    except:
        startyear, endyear = period,period
    if startyear!=endyear:
        period = "%s_%s"%(startyear, endyear)
    else:
        period = startyear
    rep = "d2v+tfidf" # sys.argv[1]## representations
    ## base dir
    base_dir = "/home/w2wei/data"
    data_dir = os.path.join(base_dir, "latest_3M_analysis")

    ## gensim dir and files
    gensim_dir = "/home/w2wei/data/gensim"
    train_corpus_file = os.path.join(gensim_dir, "medline_%s_all_docs.txt"%period)
    train_corpus_doc_pmid_index_file = os.path.join(gensim_dir, "medline_%s_all_docs_index.pkl"%period)
    gensim_model_dir = os.path.join(gensim_dir,"model")
    default_model = "doc2vec_medline_%s_def"%period# default setting
    doc2vec_model_file = os.path.join(gensim_model_dir, default_model)    
    
    ## d2v files and dirs
    d2v_base_dir = os.path.join(base_dir, "doc2vec")
    if not os.path.exists(d2v_base_dir):
        os.makedirs(d2v_base_dir)
    d2v_dir = os.path.join(d2v_base_dir, "d2v_%s"%period)
    if not os.path.exists(d2v_dir):
        os.makedirs(d2v_dir)
    train_corpus_pmid_d2v_file = os.path.join(d2v_dir, "medline_%s_pmid_d2v_dict.pkl"%period)
    
    ## tfidf files and dirs
    tfidf_base_dir = os.path.join(base_dir, "tfidf")
    tfidf_dir = os.path.join(tfidf_base_dir, "tfidf_%s"%period)
    idf_file = os.path.join(tfidf_dir, "medline_%s_idf.pkl"%period)
    train_corpus_tfidf_dict_file = os.path.join(tfidf_dir, "medline_%s_pmid_tfidf_dict.pkl"%period)

    ## test data dirs and files
    query_name = "NLM2007"
    testdata_dir = os.path.join(base_dir,"lu_data","clean")
    N2007_dir = os.path.join(testdata_dir, "NLM2007")
    N2007_tam_file = os.path.join(N2007_dir, "query_tam.pkl")
    clean_N2007_text_file = os.path.join(N2007_dir, "NLM2007_all_docs.txt") # one doc per line
    clean_N2007_docid_pmid_mapping_file = os.path.join(N2007_dir, "NLM2007_all_docs_index.pkl") # one doc per line    
    
    ## result dir and files
    knn_base_dir = os.path.join(base_dir, "knn")
    knn_dir = os.path.join(knn_base_dir,"knn_%s"%period)
    if not os.path.exists(knn_dir):
        os.makedirs(knn_dir)
    test_set_pmid_knn_dict_file = os.path.join(knn_dir, "%s_MEDLINE_%s_d2v_tfidf_knn_dict.pkl"%(query_name, period))

    ## Load D2V model
    model = load_d2v_model(doc2vec_model_file)
    print "D2V model loaded"

    ## Load test corpus
    test_corpus = load_test_corpus(N2007_tam_file,clean_N2007_text_file, clean_N2007_docid_pmid_mapping_file)
    test_corpus = list(test_corpus)
    print "test corpus loaded"

    ## infer vectors for docs in test corpus
    test_set_d2v_dict = infer_d2v(model, test_corpus, clean_N2007_docid_pmid_mapping_file)
    print "test set (%d records) Doc2Vec vector inferred"%len(test_corpus)

    ## infer tf-idf of docs in test corpus
    test_set_tfidf_dict = infer_tfidf(idf_file, test_corpus, clean_N2007_docid_pmid_mapping_file)
    print "test set (%d records) TF-IDF vector inferred"%len(test_corpus)

    ## Load training corpus
    train_corpus = load_train_corpus(train_corpus_file)
    print "train corpus loaded"

    ## Load d2v repr of docs in training corpus
    train_set_d2v_dict = load_train_d2v(model, train_corpus_doc_pmid_index_file, train_corpus_pmid_d2v_file)
    print "train set (%d records) Doc2Vec vector loaded "%len(train_set_d2v_dict)

    ## Load tfidf repr of docs in training corpus
    train_set_tfidf_dict = load_train_tfidf(idf_file, train_corpus, train_corpus_doc_pmid_index_file, train_corpus_tfidf_dict_file)
    print "train set (%d records) TF-IDF vector loaded "%len(train_set_tfidf_dict)

    ## find KNN
    find_knn(test_set_d2v_dict, test_set_tfidf_dict, train_set_d2v_dict, train_set_tfidf_dict, test_set_pmid_knn_dict_file, rep=rep)
    print "Done"

