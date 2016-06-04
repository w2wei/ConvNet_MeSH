'''
    remove non-ascii characters
    tokenize documents into sentences
    lower case
    remove punctuations
    recognize n-grams, n>1
'''
import os, time, re, string, sys
from collections import Counter, defaultdict
printable = set(string.printable)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
from new_mesh_and_entry_vocab import loadAll
raw_mesh_vocab, mesh_phrase_vocab, mesh_phrase_token_vocab, mesh_phrase_idx = loadAll() ## in mesh_phrase_vocab, tokens in phrases are connected with underscores, e.g., rna seq -> rna_seq
# mesh_phrase_dict = dict.fromkeys(mesh_phrase_vocab)
mesh_phrase_list = zip(mesh_phrase_vocab, mesh_phrase_token_vocab) # mesh_phrase_token_vocab
# mesh_phrase_list = zip(mesh_phrase_vocab, [1]*len(mesh_phrase_vocab)) # mesh_phrase_token_vocab
mesh_phrase_dict = defaultdict(None,mesh_phrase_list)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def recognizer(raw_text_dir):
    '''go thru one sentence just once'''
    for jnl in os.listdir(raw_text_dir):
        raw_jnl_dir = os.path.join(raw_text_dir,jnl)
        for doc in os.listdir(raw_jnl_dir):
            inFile = os.path.join(raw_jnl_dir, doc)
            text = file(inFile).read().lower()
            text = filter(lambda x: x in printable, text) ## remove non-ascii characters
            sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") ## tokenize documents into sentences, lower case

            for sent_idx in xrange(len(sent_tokenize_list)):
                updated_sent = [] ## a modified sentence
                sent_tokenize_list[sent_idx] = sent_tokenize_list[sent_idx].translate(replace_punctuation) ## remove all punctuation
                sent_tokenize_list[sent_idx] = TreebankWordTokenizer().tokenize(sent_tokenize_list[sent_idx])

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

def recognizer_baseline(raw_text_dir):
    '''iterate over all n-grams; go thru a sentence n times'''
    for jnl in os.listdir(raw_text_dir):
        raw_jnl_dir = os.path.join(raw_text_dir,jnl)
        for doc in os.listdir(raw_jnl_dir):
            inFile = os.path.join(raw_jnl_dir, doc)
            text = file(inFile).read().lower()
            text = filter(lambda x: x in printable, text) ## remove non-ascii characters
            sent_tokenize_list = sent_tokenize(text.strip().lower(), "english") ## tokenize documents into sentences, lower case

            for sent_idx in xrange(len(sent_tokenize_list)):
                # print "original sent"
                # print sent_tokenize_list[sent_idx]
                # print
                sent_tokenize_list[sent_idx] = sent_tokenize_list[sent_idx].translate(replace_punctuation) ## remove all punctuation
                sent_tokenize_list[sent_idx] = TreebankWordTokenizer().tokenize(sent_tokenize_list[sent_idx])

                for n in xrange(2,34): ## from bigrams to 34-grams
                    local_ngram_idx_list = []
                    local_ngram_list = []                
                    ngramList = find_ngrams(sent_tokenize_list[sent_idx], n)
                    for ngram_idx in xrange(len(ngramList)): # record the index of current n-gram
                        ngram = " ".join(ngramList[ngram_idx]) # format n-gram, e.g., (A,B)-> A_B
                        label = mesh_phrase_dict.get(ngram) # use a dictionary to check the existence of the n-gram
                        if label:
                            local_ngram_idx_list.append(ngram_idx)
                            local_ngram_list.append(ngram)
       
                    ## replace n-grams in the sentence
                    ngram_num = len(local_ngram_idx_list)
                    if ngram_num==0:
                        continue
                    for idx in xrange(ngram_num):
                        sent_tokenize_list[sent_idx][local_ngram_idx_list[idx]]=mesh_phrase_dict[local_ngram_list[idx]]
                        for more_idx in xrange(1,n):
                            sent_tokenize_list[sent_idx][local_ngram_idx_list[idx]+more_idx]=" "

                sent_tokenize_list[sent_idx] = re.sub(r"\b\d+\b", " ", " ".join(sent_tokenize_list[sent_idx]))## replace isolated digits
                # print "update sent"
                # print sent_tokenize_list[sent_idx]
                # print
                # raw_input("wait...")

if __name__=="__main__":
    # base_dir = os.getcwd()
    # data_dir = os.path.join(base_dir, "sample_data")
    # raw_mesh_file = r"/Users/w2wei/Dropbox/Research/DeepQA/shared with yupeng/d2016.bin"
    raw_pmcoa_dir = "/home/w2wei/Research/mesh/data/NLM/PMCOA_sample"
    # pmcoa_base_dir = "/home/w2wei/Research/mesh/data/deep_pmcoa"
    # pmcoa_sent_dir = os.path.join(pmcoa_base_dir,"sentences")    
    t0=time.time()
    recognizer2(raw_pmcoa_dir)
    t1=time.time()
    print "func 2 ", t1-t0

    t2=time.time()
    recognizer1(raw_pmcoa_dir)
    t3 = time.time()
    print "func 1 ", t3-t2