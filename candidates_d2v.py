'''
	This script learns distributed Doc2Vec representations of selected MEDLINE records.
    Input: clean titles and abstracts (e.g., ~/tiab_2014/*)
    Output: a D2V model
    NLP: sentence tokenized using TreeBank tokenizer, stopwords removed, single letters and non alphabetic characters removed, digits removed.
    After comparing different NLP procedures, this setting has the best performance. 
    Keeping stopwords and stemming words using Porter stemmer will lower the performance in optional evaluation.1    Created on Aug 3, 2016
    Updated on Aug 8, 2016
    @author: Wei Wei
'''

### Use Lee's method: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
import gensim, os, collections, random, time, re, string, cPickle, sys
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import multiprocessing as mp
from collections import defaultdict, Counter

class Consumer(mp.Process):
    def __init__(self,task_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" %mp.current_process()
                self.task_queue.task_done()
                break
            next_task.__call__()
            self.task_queue.task_done()
        return
    
class Task(object):
    def __init__(self,inFile,outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        '''Keep letter-digit combinations, and terms'''
        doc = []
        text = file(self.inputFile).read()
        sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") # a sentence list from doc 
        if sent_tokenize_list: # if sent_tokenize_list is not empty
            # porter_stemmer = PorterStemmer()
            for sent in sent_tokenize_list:
                words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                words = [word.strip(string.punctuation) for word in words]
                # words = [word for word in words if not word in stopwords.words("english")]
                words = [word for word in words if len(word)>1] # remove single letters and non alphabetic characters
                words = [word for word in words if re.search('[a-zA-Z]',word)] # remove digits
                # words = [porter_stemmer.stem(word) for word in words]
                doc+=words
        doc = filter(None, doc)
        fout = file(self.outputFile,"w")
        fout.write(" ".join(doc))

    def __str__(self):
        return "%s "%(self.inputFile)
            
def splitTexts(raw_doc_dir,doc_per_line_dir):
    '''Prepare sentences for gensim. One doc per line.'''
    if not os.path.exists(doc_per_line_dir):
        os.makedirs(doc_per_line_dir)

    tasks = mp.JoinableQueue()

    num_consumers = mp.cpu_count()
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    for doc in os.listdir(raw_doc_dir):
        inFile = os.path.join(raw_doc_dir,doc)
        outFile = os.path.join(doc_per_line_dir,doc)
        tasks.put(Task(inFile,outFile))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

def build_corpus(doc_per_line_dir, all_docs_file, all_doc_index_pkl):
    fout = file(all_docs_file,"w")
    doc_idx = defaultdict()
    idx = 1
    docList = os.listdir(doc_per_line_dir)
    for doc in docList:
        try:
            text = file(os.path.join(doc_per_line_dir,doc)).read()
            text.strip("\n")
            text = text+"\n"
            fout.write(text)
            doc_idx[idx] = doc.split('.txt')[0]
            idx+=1
        except Exception as e:
            print e
        if idx%1000==0:
            print "%d doc processed"%idx
    cPickle.dump(doc_idx, file(all_doc_index_pkl,"w"), protocol=cPickle.HIGHEST_PROTOCOL)

def read_corpus(fname, tokens_only=False):
    with open(fname) as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

if __name__ == '__main__':
    ## parameter
    time_span = sys.argv[1]
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    if startyear!=endyear:
        time_span = "%s_%s"%(startyear, endyear)
    else:
        time_span = startyear
    ## Directory settings
    base_dir = "/home/w2wei/data/"
    ## raw text dir
    tiab_base_dir = os.path.join(base_dir, "tiab")
    tiab_dir = os.path.join(tiab_base_dir, "tiab_%s"%time_span)

    ## gensim input, output, and model dirs and files
    gensim_dir = os.path.join(base_dir, "gensim")    
    doc2vec_input_dir = os.path.join(gensim_dir,"medline_%s_doc_per_line"%time_span) # One sentence per line, one document per file
    if not os.path.exists(doc2vec_input_dir):
        os.makedirs(doc2vec_input_dir)
    doc2vec_alldocs_file = os.path.join(gensim_dir,"medline_%s_all_docs.txt"%time_span) # One sentence per line, all sentences in one document
    doc_index_pkl = os.path.join(gensim_dir, "medline_%s_all_docs_index.pkl"%time_span) # index of all docs mapped to PMID
    
    model_dir = os.path.join(gensim_dir,"model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    ## load corpus
    if not (os.path.exists(doc2vec_alldocs_file) and os.path.exists(doc_index_pkl)):
        if not os.listdir(doc2vec_input_dir):
            splitTexts(tiab_dir,doc2vec_input_dir)
        build_corpus(doc2vec_input_dir, doc2vec_alldocs_file, doc_index_pkl)
    train_corpus = read_corpus(doc2vec_alldocs_file)
    print "train corpus ready"

    ## train a model
    model_name = "doc2vec_medline_%s_def"%time_span# default setting
    print "Model settings: embedding size 100, min word count 6, epoch 10"
    try:
        model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(model_dir, model_name))
        print "model loaded"
    except:
        # model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=10)
        model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=6, iter=10, workers=mp.cpu_count())
        model.build_vocab(train_corpus)
        model.train(train_corpus)
        model.save(os.path.join(model_dir, model_name))
        print "model ready"

    ## eval, optional
    print "Evaluate trained model..."
    train_corpus = read_corpus(doc2vec_alldocs_file)
    doc_pmid_idx = cPickle.load(file(doc_index_pkl))
    ranks = []
    print "extracting training corpus...this takes a while"
    train_corpus = list(train_corpus)
    print "train_corpus: ", len(train_corpus)
    print 'corpus ready'

    for doc_id in [1,10,15, 1000,10000,100000]: #0,1000,10000,100000,400000]:
        pmid = doc_pmid_idx.get(doc_id)
        print "index id: ", doc_id, " pmid: ",pmid
        vec = model.docvecs[doc_id]
        sims = model.docvecs.most_similar([vec], topn=len(model.docvecs))
        ranked_ids = [docid for docid, sim in sims]
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
    print "original doc vec ranks "
    print ranks   
    
    ranks = []
    for doc_id in [1,10,15, 1000,10000,100000]: #0,1000,10000,100000,400000]:
        pmid = doc_pmid_idx.get(doc_id)
        print "index id: ", doc_id, " pmid: ",pmid
        inferred_vector = model.infer_vector(train_corpus[doc_id-1].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        ranked_ids = [docid for docid, sim in sims]
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)

    print "inferred doc vec ranks "
    print ranks
    print "evaluation done"

