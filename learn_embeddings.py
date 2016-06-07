'''
    This script calls gensim to learn word embeddings from PMC OA set. All MeSH terms and entry terms 
    are recognized and replaced by special tokens. 
    Created on May 24, 2016
    Updated on May 26, 2016
    @author: Wei Wei
'''

import os, time, re, string, sys
from collections import defaultdict
printable = set(string.printable)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
from gensim import models#, similarities, corpora
import multiprocessing as mp
# os.system("taskset -p 0xff %d"%os.getpid())
from new_mesh_and_entry_vocab import loadAll
raw_mesh_vocab, mesh_phrase_vocab, mesh_phrase_token_vocab, mesh_phrase_idx = loadAll() ## in mesh_phrase_vocab, tokens in phrases are connected with underscores, e.g., rna seq -> rna_seq
# mesh_phrase_dict = dict.fromkeys(mesh_phrase_vocab)
mesh_phrase_list = zip(mesh_phrase_vocab, mesh_phrase_token_vocab) # mesh_phrase_token_vocab
# mesh_phrase_list = zip(mesh_phrase_vocab, [1]*len(mesh_phrase_vocab)) # mesh_phrase_token_vocab
mesh_phrase_dict = defaultdict(None,mesh_phrase_list)
mesh_phrase_size = len(mesh_phrase_dict) ## for the baseline method in Task.__call__


class Consumer(mp.Process):
    def __init__(self,task_queue): # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
#         self.result_queue = result_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
#         proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" %mp.current_process()
                self.task_queue.task_done()
                break
#             print "%s: %s"%(proc_name,next_task)
            next_task.__call__()
#             answer = next_task.__call__()
            self.task_queue.task_done()
#             self.result_queue.put(answer)
        return
    
class Task(object):
    def __init__(self,inFile,outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        '''tokenize sentences, lower cases, replace digits'''        
        text = file(self.inputFile).read().lower()
        text = filter(lambda x: x in printable, text) ## remove non-ascii characters
        sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") ## tokenize documents into sentences, lower case

        for sent_idx in xrange(len(sent_tokenize_list)):
            updated_sent = [] ## a modified sentence
            sent_tokenize_list[sent_idx] = sent_tokenize_list[sent_idx].translate(replace_punctuation) ## remove all punctuation
            sent_tokenize_list[sent_idx] = TreebankWordTokenizer().tokenize(sent_tokenize_list[sent_idx]) ## sent_tokenize_list[sent_idx] is a list of unigrams now

            term_idx = 0
            sentLen = len(sent_tokenize_list[sent_idx])
            while term_idx<sentLen:
                flag = 1
                curr_term = sent_tokenize_list[sent_idx][term_idx]
                if mesh_phrase_idx.get(curr_term):
                    maxPhraseLen = mesh_phrase_idx.get(curr_term) ## the maximum length of phrase starting with the current term
                    for n in xrange(maxPhraseLen,1,-1): ## iterate from n to 2
                        curr_n_gram = " ".join(sent_tokenize_list[sent_idx][term_idx:min(term_idx+n, sentLen)])
                        if mesh_phrase_dict.get(curr_n_gram):
                            updated_sent.append(mesh_phrase_dict.get(curr_n_gram))
                            term_idx+=n # move the pointer
                            flag = 0
                            break
                    if flag:
                        updated_sent.append(curr_term)
                        term_idx+=1
                else:
                    updated_sent.append(curr_term)
                    term_idx+=1
            sent_tokenize_list[sent_idx] = re.sub(r"\b\d+\b", " ", " ".join(updated_sent))## replace isolated digits

        self.__save__(sent_tokenize_list)

        ################### baseline method ##################
        # sentences = []
        # text = file(self.inputFile).read().lower()
        # text = filter(lambda x: x in printable, text) ## remove non-ascii characters
        
        # sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") ## tokenize documents into sentences, lower case
        # local_mesh_phrase_vocab = []
        # local_mesh_phrase_token_vocab = []
        
        # for idx in xrange(mesh_phrase_size):
        #     if mesh_phrase_vocab[idx] in text:
        #         local_mesh_phrase_vocab.append(mesh_phrase_vocab[idx])
        #         local_mesh_phrase_token_vocab.append(mesh_phrase_token_vocab[idx])
        # local_mesh_phrase_vocab_size = len(local_mesh_phrase_vocab)
        
        # for sent_idx in xrange(len(sent_tokenize_list)):
        #     sent_tokenize_list[sent_idx] = sent_tokenize_list[sent_idx].translate(replace_punctuation) ## remove all punctuation
        #     sent_tokenize_list[sent_idx] = " ".join(TreebankWordTokenizer().tokenize(sent_tokenize_list[sent_idx]))
        #     for voc_idx in xrange(local_mesh_phrase_vocab_size):
        #         sent_tokenize_list[sent_idx] = re.sub(r"\b%s\b"%local_mesh_phrase_vocab[voc_idx], local_mesh_phrase_token_vocab[voc_idx], sent_tokenize_list[sent_idx]) ## recognize all phrase MeSH terms
        #         sent_tokenize_list[sent_idx] = re.sub(r"\b\d+\b", " ", sent_tokenize_list[sent_idx])## replace isolated digits
        ################ end of baseline ####################    

    def __save__(self,sentences):
        fout = file(self.outputFile,"w")
        fout.write("\n".join(sentences))

    def __str__(self):
        return "%s "%(self.inputFile)

class Sentence(object):
    def __init__(self,wkdir,evalDir):
        self.wkdir = wkdir
        self.evalDir = evalDir
    
    def __iter__(self):
        for jnl in os.listdir(self.wkdir):
            jnl_dir = os.path.join(self.wkdir, jnl)
            for doc in os.listdir(jnl_dir):
                if doc not in os.listdir(self.evalDir): # if this doc is not in the evaluation set, split it for word2vec training
                    for line in file(os.path.join(jnl_dir,doc)):
                        yield line.split()
            
def splitTexts(raw_text_dir,sent_dir):
    '''Prepare sentences for gensim. One sentence per line.'''
    print "Preprocess texts..."
    tasks = mp.JoinableQueue()

    num_consumers = mp.cpu_count()#*4
    print "creating %d consumers "%num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    t0=time.time()
    for w in consumers:
        w.start()
    
    for jnl in os.listdir(raw_text_dir):
        raw_jnl_dir = os.path.join(raw_text_dir,jnl)
        jnl_dir = os.path.join(sent_dir, jnl)
        for path in [raw_jnl_dir, jnl_dir]:
            if not os.path.exists(path):
                os.mkdir(path)
        for doc in os.listdir(raw_jnl_dir):
            if doc not in os.listdir(jnl_dir): ## if parsed text does not exist
                inFile = os.path.join(raw_jnl_dir, doc)
                outFile = os.path.join(jnl_dir, doc)
                tasks.put(Task(inFile, outFile))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()
    t2=time.time()
    print "Sentence prep time: ", t2-t0
    
if __name__ == '__main__':
    ## Directory settings
    raw_pmcoa_dir = "/home/w2wei/Research/mesh/data/NLM/PMCOA"
    pmcoa_base_dir = "/home/w2wei/Research/mesh/data/deep_pmcoa"
    pmcoa_sent_dir = os.path.join(pmcoa_base_dir,"sentences")
    gensim_result_dir = os.path.join(pmcoa_base_dir, "gensim_results")
    models_dir = os.path.join(gensim_result_dir,"models")

    for path in [raw_pmcoa_dir,pmcoa_base_dir, pmcoa_sent_dir, gensim_result_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    # print "Text pre-processing.\n"
    # # if not os.listdir(pmcoa_sent_dir):
    # splitTexts(raw_pmcoa_dir, pmcoa_sent_dir) # for the BioASQ corpus, 3.1 M medline
    # print "Sentences for Word2Vec are ready."

    eval_set_dir = "/home/w2wei/Research/mesh/data/TREC/2005/4584rel/database"

    # train a Word2Vec model
    t0=time.time()
    feature_num = 100
    min_word_count = 40
    num_workers = mp.cpu_count()*2
    window = 5
    
    sentences = Sentence(pmcoa_sent_dir,eval_set_dir) # 3101418
    print "Start training..."
    t1 = time.time()
    model = models.Word2Vec(sentences,size=feature_num,window=window,min_count=min_word_count,workers=num_workers)
    t2 = time.time()
    print "Training time: ",t2-t1
    model.save(os.path.join(models_dir,"dim100_sample_1M_window_5.ml"))
    print "Model saved"
    # t1=time.time()
    # model = models.Word2Vec.load(os.path.join(models_dir,"dim100_sample_1M_window_5.ml"))
    # t2=time.time()
    # print t2-t1
    # vocab = model.vocab.keys() # vocab size 1043445
    # print "vocab size ",len(vocab)

    # fsim = file(os.path.join(bioasq_model_dir,"model_bioasq_stemmed_vocab_%d.simwd"%len(vocab)),"w")
    # for term in vocab:
    #     rec = model.most_similar(term,topn=3)
    #     rec = term+"\t"+str(rec)+"\n"
    #     fsim.write(rec)
        
    # t3=time.time()
    # print t3-t0
    
    
