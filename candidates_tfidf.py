'''
    This script computes the inversed document frequencies of terms in the vocabulary.
    Text preprocessing method:
    1. lower case
    2. only english words
    3. tokenized using nltk
    4. punctuation removed
    5. single letters and non alphabetic characters removed
    6. digits removed

    Created on July 22, 2016
    Updated on July 22, 2016
    @author: Wei Wei
'''
import os, cPickle, time, math, sys
from collections import Counter, defaultdict
import multiprocessing as mp 

class Consumer(mp.Process):
    def __init__(self,task_queue, result_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        
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
            self.result_queue.put(answer)
        return
    
class Task(object):
    def __init__(self,text):
        self.text = text
    
    def __call__(self):
        '''Keep letter-digit combinations, and terms'''
        tokens = list(set(self.text[:-1].split(' ')))
        return tokens
            
def count_df(inFile, outFile):
    '''Compute doc freq of tokens'''
    tasks = mp.JoinableQueue()
    results = mp.Queue()

    num_consumers = mp.cpu_count()
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks, results) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()

    fin = file(inFile)
    doc_ct = 0
    for doc in fin:
        doc_ct+=1
        tasks.put(Task(doc))
    print "total doc count: ", doc_ct
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

    res_ct = 0
    res_list = []

    while doc_ct:
        df = results.get()
        res_list+= df
        doc_ct-=1
        res_ct+=1

    doc_freq = Counter(res_list) 
    cPickle.dump(doc_freq, file(outFile,"w"))
    return  

def build_idf(df_file, idf_file):
    '''build inv doc freq from df'''
    df_dict = cPickle.load(file(df_file))
    doc_num = 500000 #14638795 # PubMed papers from 1995/01/01 to 2015/12/31
    log_base = 2.0
    idf_dict = defaultdict()
    for term, freq in df_dict.iteritems():
        idf_dict[term] = math.log(1.0*doc_num/freq, log_base)
    cPickle.dump(idf_dict, file(idf_file,"w"))
    return

if __name__ == '__main__':
    ## Directory settings
    time_span = sys.argv[1]
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    if startyear!=endyear:
        time_span = "%s_%s"%(startyear, endyear)
    else:
        time_span = startyear 

    base_dir = "/home/w2wei/data"
    tfidf_base_dir = os.path.join(base_dir, "tfidf")
    tfidf_dir = os.path.join(tfidf_base_dir, "tfidf_%s"%time_span)
    if not os.path.exists(tfidf_dir):
        os.makedirs(tfidf_dir)
    df_file = os.path.join(tfidf_dir, "medline_%s_doc_freq.pkl"%time_span)
    idf_file= os.path.join(tfidf_dir, "medline_%s_idf.pkl"%time_span)
    
    gensim_dir = os.path.join(base_dir, "gensim")
    doc2vec_input_file = os.path.join(gensim_dir,"medline_%s_all_docs.txt"%time_span) # One doc per line

    t0=time.time()
    if not os.path.exists(df_file):
        count_df(doc2vec_input_file, df_file)
    t1=time.time()
    print "tf-idf, count time cost: ",t1-t0
    build_idf(df_file, idf_file)
    t2 = time.time()
    print "make idf, time: ", t2-t1

  