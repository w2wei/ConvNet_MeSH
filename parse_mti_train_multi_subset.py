'''
    Parse PubMed abstracts and titles and save resulted matrices for training CNN models.
    Modified on the Severyn's code (https://github.com/aseveryn/deep-qa).
    MeSH terms are selected from KNN articles.
    Include MTI predictions as features.

    a)  Split the training set into multiple small subsets. 
    Train models on all/randomly selected subsets and test their performance on the same test set. 
    If the subset approach works, in all the following experiments train models on subsets in parallel 
    and then merge the results using a vote approach. 

    Created on August 8, 2016
    Updated on August 8, 2016
    @author: Wei Wei'''

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
from knn_data import Data
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
        raw_cand_data = raw_nbr_tam[pmid]
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

def load_all_data(raw_train, raw_dev, raw_test, stoplist):
    qids, questions, answers, labels = [], [], [], []
    obj_list = [raw_train, raw_dev, raw_test]
    train_pmids = raw_train.query_pmids
    dev_pmids = raw_dev.query_pmids
    test_pmids = raw_test.query_pmids
    pmid_groups = [train_pmids, dev_pmids, test_pmids]

    for i in range(len(obj_list)):
        pmidList = pmid_groups[i]
        for pmid in pmidList: 
            title, abstract, raw_mesh = obj_list[i].query_tam[pmid]
            raw_question = " ".join(title+abstract)
            clean_question = parser(raw_question)
            std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case
            ## select candidates from 20 nearest neighbors
            sorted_nbr_dict = sorted(obj_list[i].nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20]
            cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
            ## select candidates from 50 neighbor neighbors
            # cand_pmids = [x[0] for x in obj_list[i].nbr_dict[pmid]] ## candidates from neighbors, from all 50 nbrs
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

def load_data(raw_data, stoplist, idx):
    '''Parse raw MEDLINE records; extract PMID, title, abstract, and MeSH'''
    ## idx=0, train; idx=1, dev; idx=2, test
    qids, questions, answers, labels = [], [], [], []
    gold_std_answers = {}
    pmidList = raw_data.query_pmids
    for pmid in pmidList:
        title, abstract, raw_mesh = raw_data.query_tam[pmid]
        raw_question = " ".join(title+abstract)
        clean_question = parser(raw_question)
        std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case
        ## select candidates from 20 nearest neighbors
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20] 
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
        ## select candidates from 50 neighbor neighbors
        # cand_pmids = [x[0] for x in obj_list[i].nbr_dict[pmid]] ## candidates from neighbors, from all 50 nbrs
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
        ex = np.ones(max_sent_length) * dummy_word_idx
        for i, token in enumerate(sentence):
            idx = alphabet.get(token, UNKNOWN_WORD_IDX)
            ex[i] = idx
        data_idx.append(ex)
    data_idx = np.array(data_idx).astype('int32')
    return data_idx

def get_raw_cand_mesh(cand_pmids, raw_nbr_tam):
    '''get the occurrence count of every candidate MeSH in 20 nearest neighbors'''
    cand_mesh_list = []
    for pmid in cand_pmids:
        raw_cand_data = raw_nbr_tam[pmid]
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

def load_mti_data(raw_mti_file):
    '''collect MTI predictions for every dataset'''
    fin = file(raw_mti_file)
    pmid_mesh_dict = defaultdict()

    for line in fin:
        contents = line.split("|")
        if len(contents)==1:
            continue
        pmid, raw_mesh = contents[:2]
        raw_mesh = raw_mesh.strip("*").lower()
        clean_mesh = rawMesh_meshAndEntry_dict.get(raw_mesh)
        if clean_mesh:
            pmid_mesh_dict.setdefault(pmid, []).append(clean_mesh[0])
    return pmid_mesh_dict

def compute_mti_features(qids, questions, answers, raw_mti_file):
    '''Include the prediction from MTI as a feature'''
    pmid_mesh_dict = load_mti_data(raw_mti_file)

    q_mti_list = []
    a_mti_list = []
    for pmid, termList in zip(qids, questions):
        q_mti = np.zeros(q_max_sent_length)
        q_mti_list.append(q_mti)

    for pmid, meshList in zip(qids, answers):
        mti_mesh_list = pmid_mesh_dict.get(pmid)
        a_mti = np.zeros(a_max_sent_length)
        if mti_mesh_list!=None:
            mesh = meshList[0]
            mti_pred = mesh in mti_mesh_list
            a_mti[0] = mti_pred
        a_mti_list.append(a_mti)

    q_mti_list = np.vstack(q_mti_list).astype('int32')
    a_mti_list = np.vstack(a_mti_list).astype('int32')
    return q_mti_list, a_mti_list

def load_and_split_data(raw_data, stoplist):
    ''' Parse raw MEDLINE records; extract PMID, title, abstract, and MeSH; split data into multisubsets for data parallelism
        yield subsets.'''
    qids, questions, answers, labels = [], [], [], []
    gold_std_answers = defaultdict()
    subset_num = 10 ## determined from the raio of postive and negative examples in Lu's data

    sub_neg_dict = defaultdict() ## temporarily store neg_meshList 
    pmid_cleanquestion_dict = defaultdict() ## temporarily store qid and clean questions

    pmidList = raw_data.query_pmids
    
    for pmid in pmidList:
        title, abstract, raw_mesh = raw_data.query_tam[pmid] ## pmid: string, title: list ['title'], abstract: list ['ab']
        raw_question = " ".join(title+abstract) ## string
        clean_question = parser(raw_question) ## list ['w1','w2',...]
        pmid_cleanquestion_dict[pmid] = clean_question
        std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case ## list [m1, m2,...]

        ## select candidates from 20 nearest neighbors
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20] 
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
        cand_raw_mesh = get_cand_raw_mesh(cand_pmids, raw_data.nbr_tam)
        ## answers
        ### postive answers
        pos_meshList = list(set(std_raw_mesh)&set(cand_raw_mesh))
        for p_mesh in pos_meshList:
            answer = rawMesh_meshAndEntry_dict.get(p_mesh) # answer may contain None
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(p_mesh)] # answer is the pseudo mesh term itself                  
            answers.append(answer)
            qids.append(pmid)
            questions.append(clean_question)
            labels.append(1)

        ### negavtive answers
        #### keep all false terms from 20-NN candidates
        neg_meshList = list(set(cand_raw_mesh)-set(pos_meshList))
        random.shuffle(neg_meshList) ## shuffle negative example
        #### split neg_meshList into 10 folds
        subset_size =int(round(len(neg_meshList)*1.0/subset_num))
        for sub_idx in xrange(subset_num):
            if sub_idx == (subset_num-1):
                sub_neg_meshList = neg_meshList[sub_idx*subset_size:]
            else:
                sub_neg_meshList = neg_meshList[sub_idx*subset_size:(sub_idx+1)*subset_size]
            if not sub_neg_dict.get(pmid):
                sub_neg_dict[pmid]=[sub_neg_meshList]
            else:
                sub_neg_dict[pmid].append(sub_neg_meshList)

        gold_std_answers[pmid] = []
        for r_mesh in std_raw_mesh:
            clean_mesh = rawMesh_meshAndEntry_dict.get(r_mesh)
            if not clean_mesh:
                gold_std_answers[pmid].append(r_mesh)
            else:
                gold_std_answers[pmid].append(clean_mesh[0])

    for sub_idx in xrange(subset_num):
        sub_qids = []
        sub_questions = []
        sub_answers = []
        sub_labels = []
        for pmid in pmidList:
            sub_neg_meshList = sub_neg_dict.get(pmid)[sub_idx]
            sub_neg_mesh_num = len(sub_neg_meshList)
            for n_mesh in sub_neg_meshList:
                answer = rawMesh_meshAndEntry_dict.get(n_mesh) ## clean mesh terms and entry terms
                if not answer: # if answer==None
                    answer = [indvidual_mesh_parser(n_mesh)] # answer is the pseudo mesh term itself
                sub_answers.append(answer)
            sub_qids += [pmid]*sub_neg_mesh_num
            clean_question = pmid_cleanquestion_dict.get(pmid)
            sub_questions += [clean_question]*sub_neg_mesh_num
            sub_labels += [0]*sub_neg_mesh_num
        sub_qids += qids
        sub_questions += questions
        sub_answers += answers
        sub_labels += labels
        yield sub_qids, sub_questions, sub_answers, sub_labels, gold_std_answers

def test_load_train_data(raw_data, analysis_dir):
    '''this function loads L1000 and outputs a matrix. every row is a PMID with its mesh candidate count, postive candidate count, and negative count'''
    qids, questions, answers, labels = [], [], [], []
    pmidList = raw_data.query_pmids
    outDict = {}
    
    for pmid in pmidList:
        title, abstract, raw_mesh = raw_data.query_tam[pmid] ## pmid: string, title: list ['title'], abstract: list ['ab']
        raw_question = " ".join(title+abstract) ## string
        clean_question = parser(raw_question) ## list ['w1','w2',...]
        std_raw_mesh = mesh_parser4Lu_data(raw_mesh) ## gold standard mesh terms, raw term, lower case ## list [m1, m2,...]

        ## select candidates from 20 nearest neighbors
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20] 
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids

        ## get raw candidate mesh terms
        cand_raw_mesh = get_cand_raw_mesh(cand_pmids, raw_data.nbr_tam)
        pos_meshList = list(set(std_raw_mesh)&set(cand_raw_mesh))
        neg_meshList = list(set(cand_raw_mesh)-set(pos_meshList))

        pos_answers = []
        for p_mesh in pos_meshList:
            answer = rawMesh_meshAndEntry_dict.get(p_mesh) # answer may contain None
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(p_mesh)] # answer is the pseudo mesh term itself                  
            pos_answers.append(answer)
            qids.append(pmid)

        neg_answers = []
        for n_mesh in neg_meshList:
            answer = rawMesh_meshAndEntry_dict.get(n_mesh) ## clean mesh terms and entry terms
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(n_mesh)] # answer is the pseudo mesh term itself       
            neg_answers.append(answer)
            qids.append(pmid)
        outDict[pmid]=[pos_answers, neg_answers]
    ## output
    cPickle.dump(outDict, file(os.path.join(analysis_dir, "original_pmid_pos_neg_cands_dict.pkl"),"w"))
    outMatrix = []
    for pmid in pmidList:
        pos, neg = outDict.get(pmid)
        out = [pmid, len(pos)+len(neg), len(pos), len(neg)]
        outMatrix.append(out)
    outMatrix = np.array(outMatrix, dtype=np.float)
    np.save(os.path.join(analysis_dir, "original_pmid_pos_neg_cand_mx.npy"), outMatrix)

def test_load_and_split_data(raw_data, analysis_dir):
    ''' Get statistics from split NLM1000 subsets'''
    qids, questions, answers, labels = [], [], [], []
    gold_std_answers = defaultdict()
    subset_num = 10 ## determined from the raio of postive and negative examples in Lu's data

    sub_neg_dict = defaultdict() ## temporarily store neg_meshList 
    pos_dict = defaultdict() ## save pos mesh list of every PMID

    pmidList = raw_data.query_pmids
    for pmid in pmidList:
        title, abstract, raw_mesh = raw_data.query_tam[pmid] ## pmid: string, title: list ['title'], abstract: list ['ab']
        ## select candidates from 20 nearest neighbors
        sorted_nbr_dict = sorted(raw_data.nbr_dict[pmid],key=itemgetter(1),reverse=True)[:20] 
        cand_pmids = [x[0] for x in sorted_nbr_dict] # 20 nearest neighbor pmids
        cand_raw_mesh = get_cand_raw_mesh(cand_pmids, raw_data.nbr_tam)
        std_raw_mesh = mesh_parser4Lu_data(raw_mesh) 
        ### postive answers
        pos_meshList = list(set(std_raw_mesh)&set(cand_raw_mesh))
        local_answers = []
        for p_mesh in pos_meshList:
            answer = rawMesh_meshAndEntry_dict.get(p_mesh) # answer may contain None
            if not answer: # if answer==None
                answer = [indvidual_mesh_parser(p_mesh)] # answer is the pseudo mesh term itself                  
            local_answers.append(answer)
        pos_dict[pmid]=local_answers

        ### negavtive answers
        neg_meshList = list(set(cand_raw_mesh)-set(pos_meshList))
        #### split neg_meshList into 10 folds
        sub_neg_dict[pmid]=[]
        subset_size =int(round(len(neg_meshList)*1.0/subset_num))
        for sub_idx in xrange(subset_num):
            if sub_idx == (subset_num-1):
                sub_neg_meshList = neg_meshList[sub_idx*subset_size:]
            else:
                sub_neg_meshList = neg_meshList[sub_idx*subset_size:(sub_idx+1)*subset_size]
            if not sub_neg_dict.get(pmid):
                sub_neg_dict[pmid]=[sub_neg_meshList]
            else:
                sub_neg_dict[pmid].append(sub_neg_meshList)

        gold_std_answers[pmid] = []
        for r_mesh in std_raw_mesh:
            clean_mesh = rawMesh_meshAndEntry_dict.get(r_mesh)
            if not clean_mesh:
                gold_std_answers[pmid].append(r_mesh)
            else:
                gold_std_answers[pmid].append(clean_mesh[0])

    outMatrix_dict = {}
    subset_pmid_neg_cand_list = []
    for sub_idx in xrange(subset_num):
        sub_outmatrix = []
        subset_pmid_neg_cand_dict = {}
        for pmid in pmidList:
            sub_neg_mesh = []
            sub_neg_meshList = sub_neg_dict.get(pmid)[sub_idx]
            sub_neg_mesh_num = len(sub_neg_meshList)
            for n_mesh in sub_neg_meshList:
                answer = rawMesh_meshAndEntry_dict.get(n_mesh) ## clean mesh terms and entry terms
                if not answer: # if answer==None
                    answer = [indvidual_mesh_parser(n_mesh)] # answer is the pseudo mesh term itself
                sub_neg_mesh.append(answer)
            sub_pos_mesh = pos_dict.get(pmid)
            sub_outmatrix.append([pmid, len(sub_pos_mesh)+len(sub_neg_mesh), len(sub_pos_mesh), len(sub_neg_mesh)])
            subset_pmid_neg_cand_dict[pmid]=sub_neg_mesh
        sub_outmatrix = np.array(sub_outmatrix, dtype=np.float)
        outMatrix_dict[sub_idx] = sub_outmatrix
        subset_pmid_neg_cand_list.append(subset_pmid_neg_cand_dict)
    cPickle.dump(outMatrix_dict, file(os.path.join(analysis_dir,"subset_pmid_pos_neg_cand_mx.pkl"),"w"))
    cPickle.dump(subset_pmid_neg_cand_list, file(os.path.join(analysis_dir, "subset_pmid_neg_cand_list.pkl"),"w"))

def test_eval_split_data(analysis_dir):
    '''evaluate split data'''
    ori_dict = cPickle.load(file(os.path.join(analysis_dir, "original_pmid_pos_neg_cands_dict.pkl")))
    ori_matrix = np.load(os.path.join(analysis_dir, "original_pmid_pos_neg_cand_mx.npy"))
    sub_matrix_dict = cPickle.load(file(os.path.join(analysis_dir,"subset_pmid_pos_neg_cand_mx.pkl")))
    ## sum of # neg mesh in all subsets = sum of # neg mesh in original set. checked.
    ori_neg_num_sum = sum(ori_matrix[:,3])
    print "ori_neg_num_sum: ", ori_neg_num_sum
    sub_neg_num_list = []
    for _, mx in sub_matrix_dict.iteritems():
        sub_neg_num_list.append(sum(mx[:, 3]))
    print "sum of sum neg num: ", sum(sub_neg_num_list)
    assert ori_neg_num_sum==sum(sub_neg_num_list)
    ## pos mesh counts are consistent among subsets, checked.
    ori_pos_vec = ori_matrix[:,2]
    for idx, mx in sub_matrix_dict.iteritems():
        sub_pos_vec = mx[:,2]
        assert list(sub_pos_vec)==list(ori_pos_vec)
    ## every subset has 1000 pmids, checked.
    for idx, mx in sub_matrix_dict.iteritems():
        assert mx.shape[0]==1000
    ## sum of neg counts in every subset = all neg mesh count, checked
    neg_sum = 0
    for idx, mx in sub_matrix_dict.iteritems():
        neg_sum+=sum(mx[:,3])
    all_neg = 0
    for pmid, val in ori_dict.iteritems():
        all_neg+=len(val[1])
    assert neg_sum==all_neg
    print neg_sum
    ## given a PMID, no overlap among neg mesh from different subsets, checked
    subset_pmid_neg_cand_list = cPickle.load(file(os.path.join(analysis_dir, "subset_pmid_neg_cand_list.pkl")))
    pmidList = ori_dict.keys()
    for pmid in pmidList:
        pool = set([])
        for subset in subset_pmid_neg_cand_list:
            subset_neg_mesh = subset.get(pmid)
            subset_neg_mesh = [item for sublist in subset_neg_mesh for item in sublist]
            assert (pool&set(subset_neg_mesh))==set([])

if __name__ == '__main__':
    wkdir = sys.argv[1] 
    stoplist = None
    '''
    Input dataformat: MEDLINE records. Include title, abstract, MeSH terms, entry terms, print entry terms, etc.
    Parser extracts these entities.
    title and abstract --> query
    mesh, entry, print entry --> document
    '''
    ## raw data preparation: prepare three files, train, dev and test. ready to use. the traiing code just need files names
    data_dir = "/home/w2wei/projects/pointwiseLTR/data/knn_sample"
    raw_data_dir = os.path.join(data_dir, "raw_data")    
    train_dir = os.path.join(raw_data_dir, "L1000")
    train_mti_file = os.path.join(train_dir, "L1000_MTI.out")
    dev_dir = os.path.join(raw_data_dir, "SMALL200")
    dev_mti_file = os.path.join(dev_dir, "S200_MTI.out")
    test_dir= os.path.join(raw_data_dir, "NLM2007")
    test_mti_file = os.path.join(test_dir, "NLM2007_MTI.out")
    clean_data_dir = os.path.join(data_dir, "clean_data")

    ## pre-process the raw data to generate data objects 
    raw_train = Data(raw_data_dir, clean_data_dir)
    raw_train.large1000()
    raw_dev = Data(raw_data_dir, clean_data_dir)    
    raw_dev.small200()
    raw_test = Data(raw_data_dir, clean_data_dir)    
    raw_test.nlm2007()
    
    ## Load MTI predictions for train/dev/test
    outdir = os.path.join(data_dir, wkdir)#'{}'.format(name.upper())
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ## for test only, deleted it
    # qids, questions, answers, labels = load_all_data(raw_dev, raw_dev, raw_dev, stoplist) ## load data from sample_merged
    # alphabet = Alphabet(start_feature_id=0) ## a dictionary {token: token index}
    # alphabet.add('UNKNOWN_WORD_IDX')
    # add_to_vocab(answers, alphabet)
    # add_to_vocab(questions, alphabet)
    # dummy_word_idx = alphabet.fid
    # print "vocabulary loaded, size ", alphabet.fid    
    ## end of test

    qids, questions, answers, labels = load_all_data(raw_train, raw_dev, raw_test, stoplist) ## load data from sample_merged
    print "all qids: ", len(qids), len(set(qids))
    ## Build a vocabulary
    alphabet = Alphabet(start_feature_id=0) ## a dictionary {token: token index}
    alphabet.add('UNKNOWN_WORD_IDX')
    add_to_vocab(answers, alphabet)
    add_to_vocab(questions, alphabet)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
    dummy_word_idx = alphabet.fid
    print "vocabulary loaded, size ", alphabet.fid

    ## Get max sentence length in questions and answers
    q_max_sent_length = max(map(lambda x: len(x), questions))
    a_max_sent_length = max(map(lambda x: len(x), answers))
    print 'q_max_sent_length', q_max_sent_length
    print 'a_max_sent_length', a_max_sent_length

    ## Convert training data and split them
    train_data_dir = os.path.join(outdir, "train")
    analysis_dir = os.path.join(outdir, "analysis")
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    train_data = load_and_split_data(raw_train, stoplist)
    ## For code validation
    # test_load_train_data(raw_train, analysis_dir) ## generate statistics from training data
    # test_load_and_split_data(raw_train, analysis_dir)
    # test_eval_split_data(analysis_dir)
    ## End 
    
    sub_idx = 0
    for data in train_data:
        sub_idx+=1
        qids, questions, answers, labels, std_mesh_dict = data
        cand_answer_tokens = [ans[0] for ans in answers] ## save answer_tokens
        candFreqDict = load_knn_rel_data(raw_train)

        qids = np.array(qids)
        labels = np.array(labels).astype('int32')

        _, counts = np.unique(labels, return_counts=True) ## counts of unique components in label array

        q_knn_counts, a_knn_counts = compute_knn_features(qids, questions, answers, candFreqDict)
        q_mti_pred, a_mti_pred = compute_mti_features(qids, questions, answers, train_mti_file)

        q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)

        questions_idx = convert2indices(questions, alphabet, dummy_word_idx, q_max_sent_length)
        answers_idx = convert2indices(answers, alphabet, dummy_word_idx, a_max_sent_length)

        basename = "train"
        sub_dir = os.path.join(train_data_dir, "sub_%d"%sub_idx)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        np.save(os.path.join(sub_dir, '{}.qids.npy'.format(basename)), qids)
        np.save(os.path.join(sub_dir, '{}.questions.npy'.format(basename)), questions_idx)
        np.save(os.path.join(sub_dir, '{}.answers.npy'.format(basename)), answers_idx)
        np.save(os.path.join(sub_dir, '{}.labels.npy'.format(basename)), labels)
        np.save(os.path.join(sub_dir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
        np.save(os.path.join(sub_dir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)
        np.save(os.path.join(sub_dir, '{}.q_knn_counts.npy'.format(basename)), q_knn_counts)
        np.save(os.path.join(sub_dir, '{}.a_knn_counts.npy'.format(basename)), a_knn_counts)
        np.save(os.path.join(sub_dir, '{}.q_mti.npy'.format(basename)), q_mti_pred)
        np.save(os.path.join(sub_dir, '{}.a_mti.npy'.format(basename)), a_mti_pred)
        cPickle.dump(cand_answer_tokens, file(os.path.join(sub_dir, '{}.cand_mesh.pkl'.format(basename)),'w'))
        cPickle.dump(std_mesh_dict, file(os.path.join(sub_dir, '{}.std_mesh.pkl'.format(basename)),'w'))
        print "%s data saved."%basename

    ## Convert dev and test sets
    obj_list = [raw_dev, raw_test]
    mti_file_list = [dev_mti_file, test_mti_file]
    name_list = ['dev','test']
    for i in range(len(obj_list)):
        qids, questions, answers, labels, std_mesh_dict = load_data(obj_list[i], stoplist, i)
        cand_answer_tokens = [ans[0] for ans in answers] ## save answer_tokens
        candFreqDict = load_knn_rel_data(obj_list[i])

        qids = np.array(qids)
        labels = np.array(labels).astype('int32')

        _, counts = np.unique(labels, return_counts=True) ## counts of unique components in label array

        q_knn_counts, a_knn_counts = compute_knn_features(qids, questions, answers, candFreqDict)
        q_mti_pred, a_mti_pred = compute_mti_features(qids, questions, answers, mti_file_list[i])

        q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)

        questions_idx = convert2indices(questions, alphabet, dummy_word_idx, q_max_sent_length)
        answers_idx = convert2indices(answers, alphabet, dummy_word_idx, a_max_sent_length)

        basename = name_list[i]
        np.save(os.path.join(outdir, '{}.qids.npy'.format(basename)), qids)
        np.save(os.path.join(outdir, '{}.questions.npy'.format(basename)), questions_idx)
        np.save(os.path.join(outdir, '{}.answers.npy'.format(basename)), answers_idx)
        np.save(os.path.join(outdir, '{}.labels.npy'.format(basename)), labels)
        np.save(os.path.join(outdir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
        np.save(os.path.join(outdir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)
        np.save(os.path.join(outdir, '{}.q_knn_counts.npy'.format(basename)), q_knn_counts)
        np.save(os.path.join(outdir, '{}.a_knn_counts.npy'.format(basename)), a_knn_counts)
        np.save(os.path.join(outdir, '{}.q_mti.npy'.format(basename)), q_mti_pred)
        np.save(os.path.join(outdir, '{}.a_mti.npy'.format(basename)), a_mti_pred)
        cPickle.dump(cand_answer_tokens, file(os.path.join(outdir, '{}.cand_mesh.pkl'.format(basename)),'w'))
        cPickle.dump(std_mesh_dict, file(os.path.join(outdir, '{}.std_mesh.pkl'.format(basename)),'w'))
        print "%s data saved."%basename