'''
    Exp 46
    Training set: 5K from MEDLINE 1997
    Dev set: 500 from MEDLINE 1997
    Test set: NLM2007
'''

import os, sys, string, time, re, random, subprocess, cPickle
from collections import defaultdict, Counter
from operator import itemgetter
from alphabet import Alphabet
from corpus import loadCorpus
from collections import defaultdict
import numpy as np
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
printable = set(string.printable)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
from new_mesh_and_entry_vocab import loadAll
raw_mesh_entry_vocab, mesh_phrase_idx, mesh_phrase_dict, rawMesh_meshAndEntry_dict = loadAll() ## in mesh_phrase_vocab, tokens in phrases are connected with underscores, e.g., rna seq -> rna_seq
raw_mesh_vocab = rawMesh_meshAndEntry_dict.keys()
clean_mesh_vocab = [x[0] for x in rawMesh_meshAndEntry_dict.values()]
raw_mesh_set = set(raw_mesh_vocab)
clean_mesh_set = set(clean_mesh_vocab)
import candidates_load_data as cdata 
import knn_data as kdata
import knn_data2 as kdata2
from pprint import pprint

UNKNOWN_WORD_IDX = 0


print "All modules loaded"

def parser(inText):
    text = filter(lambda x: x in printable, inText) ## remove non-ascii characters
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
    tokenList = " ".join(sent_tokenize_list).split()
    return tokenList

def mesh_parser(termList):
    newTermList = []
    for term in termList:
        text = filter(lambda x: x in printable, term) 
        text = text.translate(replace_punctuation)
        text = " ".join(text.split())
        newTermList.append(text)
    return newTermList

def indvidual_mesh_parser(term):
    text = filter(lambda x: x in printable, term) 
    text = text.translate(replace_punctuation)
    text = " ".join(text.split()).lower()
    text = text.replace(" ", "_")
    return text

def mesh_parser4Lu_data(text):
    text = re.sub("[-*&]"," ",text)
    mhList = text.split("!")
    mhList = [mh.strip("*") for mh in mhList]
    return mhList

def get_cand_raw_mesh(cand_pmids, raw_nbr_tam):
    cand_mesh_list = []
    for pmid in cand_pmids:
        raw_cand_data = raw_nbr_tam.get(pmid)
        if not raw_cand_data:
            continue
        if len(raw_cand_data)==3:
            raw_cand_mesh = raw_cand_data[2]
        else:
            possible_raw_cand_mesh = raw_cand_data[-1]
            if isinstance(possible_raw_cand_mesh, str):
                raw_cand_mesh = possible_raw_cand_mesh
            else:
                raw_cand_mesh = ''
        cand_mesh = mesh_parser4Lu_data(raw_cand_mesh)
        cand_mesh = filter(None, cand_mesh) # remove ''
        cand_mesh_list+=cand_mesh
    cand_mesh_list = list(set(cand_mesh_list))
    cand_mesh_list.sort()
    return cand_mesh_list

def get_raw_cand_mesh(cand_pmids, raw_nbr_tam):
    ''' For KNN occurrence count. Different from get_cand_raw_mesh. Doesn't apply set
        get the occurrence count of every candidate MeSH in 20 nearest neighbors'''
    cand_mesh_list = []
    for pmid in cand_pmids:
        raw_cand_data = raw_nbr_tam.get(pmid)
        if not raw_cand_data:
            continue
        if len(raw_cand_data)==3:
            raw_cand_mesh = raw_cand_data[2]
        else:
            possible_raw_cand_mesh = raw_cand_data[-1]
            if isinstance(possible_raw_cand_mesh, str):
                raw_cand_mesh = possible_raw_cand_mesh
            else:
                raw_cand_mesh = ''
        cand_mesh = mesh_parser4Lu_data(raw_cand_mesh)
        cand_mesh_list+=cand_mesh
    return cand_mesh_list 

def load_all_data(raw_train, raw_dev, raw_test, stoplist, train_sample_size, dev_sample_size):
    qids, questions, answers, labels = [], [], [], []
    obj_list = [raw_train, raw_dev, raw_test]
    name_list = ['train','dev','test']
    train_pmids = raw_train.query_pmids[:train_sample_size]
    dev_pmids = raw_dev.query_pmids[:dev_sample_size]
    test_pmids = raw_test.query_pmids
    pmid_groups = [train_pmids, dev_pmids, test_pmids]

    for i in range(len(obj_list)):
        pmidList = pmid_groups[i]
        print name_list[i], " pmid num ", len(pmidList)
        for pmid in pmidList: 
            title, abstract, raw_mesh = obj_list[i].query_tam[pmid]
            raw_question = " ".join(title+abstract)
            clean_question = parser(raw_question)
            std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case
            ## select candidates from 20 nearest neighbors
            sorted_nbr_dict = sorted(obj_list[i].nbr_dict[pmid],key=itemgetter(1),reverse=True)[:21]
            cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
            # cand_pmids = [lambda x:x!=pmid for x in cand_pmids]
            cand_pmids = filter(lambda x:x!=pmid, cand_pmids)
            cand_pmids = cand_pmids[:20] 
            ## select candidates from 50 neighbor neighbors
            cand_raw_mesh = get_cand_raw_mesh(cand_pmids, obj_list[i].nbr_tam)
            ## answers
            pos_meshList = list(set(std_raw_mesh)&set(cand_raw_mesh))
            ## keep all false terms in candidates
            neg_meshList = list(set(cand_raw_mesh)-set(pos_meshList)) 
            # if i!=2:
                # neg_meshList = random.sample(neg_meshList, len(pos_meshList)) # make a neg mesh subset 
            meshNum = len(pos_meshList)+len(neg_meshList) ## answer number for this PMID

            for p_mesh in pos_meshList:
                answer = rawMesh_meshAndEntry_dict.get(p_mesh) # answer may contain None
                if not answer: # if answer==None
                    answer = [p_mesh] # answer is the pseudo mesh term itself
                answers.append(answer)
                qids.append(pmid)
                questions.append(clean_question)
                labels.append(1)

            for n_mesh in neg_meshList:
                answer = rawMesh_meshAndEntry_dict.get(n_mesh) ## clean mesh terms and entry terms
                if not answer: # if answer==None
                    answer = [n_mesh] # answer is the pseudo mesh term itself            
                answers.append(answer) #  answer may contain None
                qids.append(pmid)
                questions.append(clean_question)
                labels.append(0)

    return qids, questions, answers, labels        

def load_data(raw_data, stoplist, idx, size):
    '''Parse raw MEDLINE records; extract PMID, title, abstract, and MeSH'''
    ## idx=0, train; idx=1, dev; idx=2, test
    qids, questions, answers, labels = [], [], [], []
    gold_std_answers = {}
    pmidList = raw_data.query_pmids[:size]
    for pmid in pmidList:
        title, abstract, raw_mesh = raw_data.query_tam[pmid]
        raw_question = " ".join(title+abstract)
        clean_question = parser(raw_question)
        std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case
        ## select candidates from 20 nearest neighbors
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:21] 
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
        cand_pmids = filter(lambda x:x!=pmid, cand_pmids)
        cand_pmids = cand_pmids[:20]
        ## select candidates from 50 neighbor neighbors
        cand_raw_mesh = get_cand_raw_mesh(cand_pmids, raw_data.nbr_tam)
        ## answers
        pos_meshList = list(set(std_raw_mesh)&set(cand_raw_mesh))
        ## keep all false terms from 20-NN candidates
        neg_meshList = list(set(cand_raw_mesh)-set(pos_meshList))

        meshNum = len(pos_meshList)+len(neg_meshList) ## answer number for this PMID

        for p_mesh in pos_meshList:
            answer = rawMesh_meshAndEntry_dict.get(p_mesh) # answer may contain None
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(p_mesh)] # answer is the pseudo mesh term itself                  
            answers.append(answer)
            qids.append(pmid)
            questions.append(clean_question)
            labels.append(1)

        for n_mesh in neg_meshList:
            answer = rawMesh_meshAndEntry_dict.get(n_mesh) ## clean mesh terms and entry terms
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(n_mesh)] # answer is the pseudo mesh term itself       
            answers.append(answer) #  answer may contain None
            qids.append(pmid)
            questions.append(clean_question)
            labels.append(0)

        gold_std_answers[pmid] = []
        for r_mesh in std_raw_mesh:
            clean_mesh = rawMesh_meshAndEntry_dict.get(r_mesh)
            if not clean_mesh:
                gold_std_answers[pmid].append(r_mesh)
            else:
                gold_std_answers[pmid].append(clean_mesh[0])

    return qids, questions, answers, labels, gold_std_answers
 
def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
    '''the overlap percentage of tokens between evey pair of question and answer'''
    word2df = word2df if word2df else {}
    stoplist = stoplist if stoplist else set()
    feats_overlap = []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))
        df_overlap = 0.0
        for w in word_overlap:
            df_overlap += word2df[w]
        df_overlap /= (len(q_set) + len(a_set))

        feats_overlap.append(np.array([
                         overlap, ## normalized number of overlap tokens
                         df_overlap, ## normalized df of overlap tokens
                         ]))
    return np.array(feats_overlap)

def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
    stoplist = stoplist if stoplist else []
    feats_overlap = []
    q_indices, a_indices = [], []
    for question, answer in zip(questions, answers):
        q_set = set([q for q in question if q not in stoplist])
        a_set = set([a for a in answer if a not in stoplist])
        word_overlap = q_set.intersection(a_set)
        q_idx = np.ones(q_max_sent_length) * 2
        for i, q in enumerate(question):
            value = 0
            if q in word_overlap:
                value = 1
            q_idx[i] = value
        q_indices.append(q_idx)

        a_idx = np.ones(a_max_sent_length) * 2
        for i, a in enumerate(answer):
            value = 0
            if a in word_overlap:
                value = 1
            a_idx[i] = value
        a_indices.append(a_idx)

    q_indices = np.vstack(q_indices).astype('int32')
    a_indices = np.vstack(a_indices).astype('int32')
    return q_indices, a_indices

def compute_dfs(docs):
    '''compute document frequencies. not inversed df '''
    word2df = defaultdict(float)
    for doc in docs:
        if not doc: # if the doc is None
            continue        
        for w in set(doc):
            word2df[w] += 1.0
    num_docs = len(docs)
    for w, value in word2df.iteritems():
        word2df[w] /= np.math.log(num_docs / value)
    return word2df

def add_to_vocab(data, alphabet):
    for sentence in data:
        for token in sentence:
            alphabet.add(token)

def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=40):
    data_idx = []
    for sentence in data:
        ex = np.ones(max_sent_length) * dummy_word_idx # ex is a list of token index of a sentence
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, UNKNOWN_WORD_IDX)
            ex[i] = idx
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    return data_idx

def get_clean_cand_mesh(raw_cand_mesh):
    '''process raw mesh terms, connect tokens in phrases using _'''
    mesh_list = []
    for raw_mesh in raw_cand_mesh:
        # if raw_mesh in checktags: ## remove checktags
        #     pass
        clean_mesh = rawMesh_meshAndEntry_dict.get(raw_mesh)
        if clean_mesh:
            mesh_list.append(clean_mesh[0])
        else:
            ## missing terms are entry terms, not MeSH terms
            mesh_list.append(raw_mesh)
    return mesh_list
            
def load_knn_rel_data(raw_data):
    '''Parse raw MEDLINE records; collect candidates from 20 nearest neighbors. record occurrences.'''
    pmidList = raw_data.query_pmids # query pmids
    resultDict = {}
    for pmid in pmidList:
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20] ## keep top 20 nbrs
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
        ## get raw mesh terms from all candidates
        raw_cand_mesh = get_raw_cand_mesh(cand_pmids, raw_data.nbr_tam) # raw mesh from 20 nbrs
        raw_cand_mesh = filter(None, raw_cand_mesh)
        clean_cand_mesh = get_clean_cand_mesh(raw_cand_mesh)
        cand_mesh_freq = Counter(clean_cand_mesh)
        resultDict[pmid]=cand_mesh_freq
    return resultDict

def compute_knn_features(qids, questions, answers, candFreqDict):
    '''Record the number of occurrences of every candidate in the neighbors'''
    q_freq_list = []
    a_freq_list = []
    for pmid, termList in zip(qids, questions):
        q_freq = np.zeros(q_max_sent_length)
        q_freq_list.append(q_freq)

    for pmid, meshList in zip(qids, answers):
        mesh = meshList[0]
        mesh_freq = candFreqDict[pmid][mesh]
        a_freq = np.zeros(a_max_sent_length)
        a_freq[0] = mesh_freq
        a_freq_list.append(a_freq)

    q_freq_list = np.vstack(q_freq_list).astype('int32')
    a_freq_list = np.vstack(a_freq_list).astype('int32')
    return q_freq_list, a_freq_list

if __name__ == '__main__':
    '''
    Input dataformat: MEDLINE records. Include title, abstract, MeSH terms, entry terms, print entry terms, etc.
    Parser extracts these entities.
    title and abstract --> query
    mesh, entry, print entry --> document
    '''    
    wkdir = sys.argv[1] # Exp_48
    
    # time_span = sys.argv[2] # "1960_2009" 
    # try:
    #     startyear, endyear = time_span.split("-")
    # except:
    #     startyear, endyear = time_span,time_span

    # time_span = "%s_%s"%(startyear, endyear)

    train_sample_size = sys.argv[2]
    dev_sample_size = sys.argv[3]

    stoplist = None

    exp_base_dir = "/home/w2wei/exp/" ## output data dir
    outdir = os.path.join(exp_base_dir, wkdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_dir = "/home/w2wei/data" ## source data dir
    lu_data_dir = os.path.join(data_dir, "lu_data")
    clean_lu_data_dir = os.path.join(data_dir, "clean")
    my_data_dir = os.path.join(data_dir, "my_data")

    ## pre-process the raw data to generate data objects 
    raw_test = kdata.Data(lu_data_dir, clean_lu_data_dir)    
    raw_test.nlm2007()
    rm_list = raw_test.query_pmids
    print "raw test ready"    
    raw_train = kdata2.Data(os.path.join(my_data_dir, "NLM5K"), 10000, rm_list)
    raw_train.run()
    rm_list = list(set(raw_test.query_pmids)|set(raw_train.query_pmids))
    print "raw train ready"
    raw_dev = kdata2.Data(os.path.join(my_data_dir, "NLM500"), 1000, rm_list)
    raw_dev.run()
    print "raw dev ready"
    print "====================="
    print "train pmid: ", len(raw_train.query_pmids)
    print "dev pmid: ", len(raw_dev.query_pmids)
    print "test pmid: ", len(raw_test.query_pmids)
    print "train & dev: ", len(set(raw_train.query_pmids)&set(raw_dev.query_pmids))
    print "train & test: ", len(set(raw_train.query_pmids)&set(raw_test.query_pmids))
    print "dev & test: ", len(set(raw_dev.query_pmids)&set(raw_test.query_pmids))
    raw_input('...')
    t0=time.time()
    qids, questions, answers, labels = load_all_data(raw_train, raw_dev, raw_test, stoplist, int(train_sample_size), int(dev_sample_size)) ## load data from sample_merged
    t1=time.time()
    print "Load all datasets: ",t1-t0 # 55 secs for 817 documents, 22689 answers
    print "qids: ", len(qids), len(set(qids))
    ### Compute document frequencies.
    seen = set() ## set of unique qids
    unique_questions = [] ## list of unique questions
    for q, qid in zip(questions, qids):
      if qid not in seen:
        seen.add(qid)
        unique_questions.append(q)

    docs = answers + unique_questions ## a list of lists # None exists in docs

    word2dfs = compute_dfs(docs) ## idf of every word
    #########

    alphabet = Alphabet(start_feature_id=0) ## a dictionary {token: token index}
    alphabet.add('UNKNOWN_WORD_IDX')

    add_to_vocab(answers, alphabet)
    add_to_vocab(questions, alphabet)

    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))

    dummy_word_idx = alphabet.fid
    print "dummy_word_idx ", dummy_word_idx

    q_max_sent_length = max(map(lambda x: len(x), questions))
    a_max_sent_length = max(map(lambda x: len(x), answers))
    print 'q_max_sent_length', q_max_sent_length
    print 'a_max_sent_length', a_max_sent_length

    obj_list = [raw_train, raw_dev, raw_test]
    name_list = ['train','dev','test']

    for i in range(len(obj_list)):
        print "name: ", name_list[i]
        if name_list[i]=='train':
            size = int(train_sample_size)
        if name_list[i]=='dev':
            size = int(dev_sample_size)
        if name_list[i]=='test':
            size = 200
        qids, questions, answers, labels, std_mesh_dict = load_data(obj_list[i], stoplist, i, size)
        cand_answer_tokens = [ans[0] for ans in answers] ## save answer_tokens
        candFreqDict = load_knn_rel_data(obj_list[i])

        qids = np.array(qids)
        labels = np.array(labels).astype('int32')

        _, counts = np.unique(labels, return_counts=True) ## counts of unique components in label array

        q_knn_counts, a_knn_counts = compute_knn_features(qids, questions, answers, candFreqDict)

        q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)

        questions_idx = convert2indices(questions, alphabet, dummy_word_idx, q_max_sent_length)
        answers_idx = convert2indices(answers, alphabet, dummy_word_idx, a_max_sent_length)

        # basename, _ = os.path.splitext(os.path.basename(fname))
        basename = name_list[i]
        print basename
        print "basename ", basename
        print os.path.join(outdir, '{}.qids.npy'.format(basename))
        print qids.shape
        print os.path.join(outdir, '{}.questions.npy'.format(basename))
        print questions_idx.shape
        print os.path.join(outdir, '{}.answers.npy'.format(basename))
        print answers_idx.shape
        print os.path.join(outdir, '{}.labels.npy'.format(basename))
        print labels.shape
        print os.path.join(outdir, '{}.q_overlap_indices.npy'.format(basename))
        print q_overlap_indices.shape
        print os.path.join(outdir, '{}.a_overlap_indices.npy'.format(basename))
        print a_overlap_indices.shape
        print os.path.join(outdir, '{}.q_knn_counts.npy'.format(basename))
        print q_knn_counts.shape
        print os.path.join(outdir, '{}.a_knn_counts.npy'.format(basename))
        print a_knn_counts.shape
        print os.path.join(outdir, '{}.cand_mesh.pkl'.format(basename))
        print len(cand_answer_tokens), cand_answer_tokens[:5]
        print os.path.join(outdir, '{}.std_mesh.pkl'.format(basename))
        print len(std_mesh_dict), std_mesh_dict.items()[:2]
        # raw_input("wait...")

        np.save(os.path.join(outdir, '{}.qids.npy'.format(basename)), qids)
        np.save(os.path.join(outdir, '{}.questions.npy'.format(basename)), questions_idx)
        np.save(os.path.join(outdir, '{}.answers.npy'.format(basename)), answers_idx)
        np.save(os.path.join(outdir, '{}.labels.npy'.format(basename)), labels)
        np.save(os.path.join(outdir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
        np.save(os.path.join(outdir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)
        np.save(os.path.join(outdir, '{}.q_knn_counts.npy'.format(basename)), q_knn_counts)
        np.save(os.path.join(outdir, '{}.a_knn_counts.npy'.format(basename)), a_knn_counts)
        cPickle.dump(cand_answer_tokens, file(os.path.join(outdir, '{}.cand_mesh.pkl'.format(basename)),'w'))
        cPickle.dump(std_mesh_dict, file(os.path.join(outdir, '{}.std_mesh.pkl'.format(basename)),'w'))

        print "saving %s data..."%basename


