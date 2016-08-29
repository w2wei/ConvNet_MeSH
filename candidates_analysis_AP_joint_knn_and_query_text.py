'''
    Recognize MeSH terms, entry terms, print entry terms as candidates from the text to be indexed.

    Load the mesh/entry_term vocabulary
    1. From d2016.bin, build {Entry_term:MeSH}, {MeSH:UI}, {MeSH:[Entry term list]}, {UI:[Entry term list]}

    For every text to be indexed, screen it in parallel with KNN and recognize mesh/entry_terms.

    Created on July 30, 2016
    Updated on July 30, 2016
    @author: Wei Wei
'''

import os, cPickle, string, time, re
from collections import defaultdict
from knn_data import Data
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import multiprocessing as mp

def build_entry_term_mesh_dict(raw_mesh_file, entry_term_mesh_dict_file):
    try:
        entry_term_mesh_dict = cPickle.load(file(entry_term_mesh_dict_file))
    except:
        raw_text = filter(None, file(raw_mesh_file).read().split("\n\n"))
        entry_term_mesh_dict = defaultdict()
        for rec in raw_text:
            mesh = ''
            entry = ''
            lines = rec.split("\n")
            for line in lines:
                if line.startswith('MH = '):
                    mesh = line.split('MH = ')[1]
                if line.startswith('ENTRY = '):
                    entry = line.split('ENTRY = ')[1]
                    if "|" in entry:
                        entry = entry.split("|")[0]
                    entry_term_mesh_dict[entry]=mesh
                if line.startswith('PRINT ENTRY = '):
                    entry = line.split('PRINT ENTRY = ')[1]
                    if "|" in entry:
                        entry = entry.split("|")[0]
                    entry_term_mesh_dict[entry]=mesh
        cPickle.dump(entry_term_mesh_dict, file(entry_term_mesh_dict_file,"w"))

def build_clean_entry_term_mesh_dict(entry_term_mesh_dict_file, clean_entry_term_mesh_dict_file):
    try:
        clean_entry_term_mesh_dict = cPickle.load(file(clean_entry_term_mesh_dict_file))
    except:    
        entry_term_mesh_dict = cPickle.load(file(entry_term_mesh_dict_file)) # {entry term: mesh}
        clean_entry_term_mesh_dict = defaultdict()
        clean_mesh_list = []
        for entry, mesh in entry_term_mesh_dict.iteritems():
            entryList = TreebankWordTokenizer().tokenize(entry.lower())
            entryList = [word.strip(string.punctuation) for word in entryList]
            entryList = filter(None, entryList)
            clean_entry = " ".join(entryList)
            clean_entry_term_mesh_dict[clean_entry]=mesh    
        cPickle.dump(clean_entry_term_mesh_dict, file(clean_entry_term_mesh_dict_file,"w"))

def build_mesh_vocab(raw_mesh_file, mesh_dict_file):
    try:
        mesh_dict = cPickle.load(file(mesh_dict_file))
    except:
        raw_text = file(raw_mesh_file)
        mesh_dict = defaultdict()
        for line in raw_text:
            if line.startswith('MH = '):
                mesh = line.split('MH = ')[1]
                mesh = mesh.split("\n")[0]
                mesh_dict[mesh] = mesh
        cPickle.dump(mesh_dict, file(mesh_dict_file,"w"))

def build_clean_mesh_vocab(mesh_dict_file, clean_mesh_dict_file):
    try:
        clean_mesh_dict = cPickle.load(file(clean_mesh_dict_file))
    except:
        mesh_dict = cPickle.load(file(mesh_dict_file)) ## mesh vocab
        clean_mesh_dict = defaultdict()
        for mesh in mesh_dict.keys():
            meshList = TreebankWordTokenizer().tokenize(mesh.lower())
            meshList = [word.strip(string.punctuation) for word in meshList]
            meshList = filter(None, meshList)
            clean_mesh = " ".join(meshList)
            clean_mesh_dict[clean_mesh]=mesh    
        cPickle.dump(clean_mesh_dict, file(clean_mesh_dict_file,"w"))

def index_ngrams(clean_entry_term_mesh_dict_file, clean_mesh_dict_file, mesh_and_entry_token_index_file):
    '''index terms in cleaned mesh terms and entry terms'''
    if not os.path.exists(mesh_and_entry_token_index_file):
        clean_entry_term_mesh_dict = cPickle.load(file(clean_entry_term_mesh_dict_file))
        clean_mesh_dict = cPickle.load(file(clean_mesh_dict_file))
        token_idx = defaultdict()
        for entry in clean_entry_term_mesh_dict.keys():
            tokens = entry.split(" ")
            for tk in tokens:
                if not token_idx.get(tk):
                    token_idx[tk]=[entry]
                else:
                    token_idx[tk].append(entry)
        for c_mesh in clean_mesh_dict.keys():
            tokens = c_mesh.split(" ")
            for tk in tokens:
                if not token_idx.get(tk):
                    token_idx[tk]=[c_mesh]
                else:
                    token_idx[tk].append(c_mesh)
        for token, mesh in token_idx.iteritems():
            token_idx[token] = list(set(mesh))
        cPickle.dump(token_idx, file(mesh_and_entry_token_index_file, "w"))

class Lu_query_on_medline1997(object):
    '''Use Lu's data as queries.'''
    def __init__(self, data_dir, resource_dir):
        self.dataDir = data_dir
        self.dictDir = resource_dir ## resources from NLM, organized in dict format
        self.period = "1997"
        self.analyDir = os.path.join(self.dataDir, "latest_3M_analysis") ## dir for mesh terms
        self.mesh_1997_dir = os.path.join(self.analyDir, "mesh_%s"%self.period) ## from corpus_medline_mesh_coverage.py
        self.nlm2007_knn_pmid_from_1997_corpus_file = os.path.join(self.analyDir, "NLM2007_as_query_knn_from_%s.pkl"%self.period) ## from corpus_medline_mesh_coverage.py
        self.nlm2007_qpmid_mesh_from_query_file = os.path.join(self.analyDir,"NLM2007_q_pmid_mesh_from_query_%s.pkl"%self.period)

        self.query_dict = defaultdict() ## {pmid:title+abstract text in raw format}
        self.std_mesh_dict = defaultdict() 
        self.knn_mesh_dict = defaultdict() ## {query pmid: knn mesh}
        self.pmid_matched_mesh_dict = defaultdict() ## matched mesh terms in every query
        self.query_coverage_list = [] ## list for the coverage of query text, i.e., mesh terms recognized from query texts
        self.knn_coverage_list = [] ## list for the coverage of knn pmids
        self.joint_coverage_list = []

    def run(self):
        self.load_query() ## {pmid:title+abstract text in raw format}
        self.screen_query()
        self.comp_query_text_coverage()
        self.load_candidates_pmids_from_KNN()
        self.load_prc_candidates()
        self.comp_coverage_joint_candidates()

    def load_query(self):
        '''load lu's data from knn_data'''
        raw_data_dir = os.path.join(data_dir,"lu_data")
        clean_dir = os.path.join(data_dir, "lu_data", "clean")

        NLM2007 = Data(raw_data_dir, clean_dir)
        NLM2007.nlm2007()
        NLM2007_pmids = NLM2007.query_pmids

        for pmid in NLM2007_pmids:
            title, abstract, _ = NLM2007.query_tam[pmid]
            self.query_dict[pmid] = ". ".join(title + abstract)

    def screen_query(self):
        '''screen queries and identify mesh candidates'''
        ## load resources
        clean_token_index = cPickle.load(file(os.path.join(self.dictDir, "mesh_and_entry_terms_token_index.pkl"))) ## a dict, {token:[clean mesh or entry]}
        clean_entry_term_mesh_dict = cPickle.load(file(os.path.join(self.dictDir, "clean_entry_term_mesh_dict.pkl")))
        clean_mesh_dict = cPickle.load(file(os.path.join(self.dictDir, "clean_mesh_dict.pkl")))  
        ## setting 1: lower case, remove punctuation, normalize words
        ### prepare clean entry terms and mesh terms
        for pmid, raw_query in self.query_dict.iteritems():
            match_dict = defaultdict() ## match terms

            ### clean this query
            clean_query = []
            sent_tokenize_list = sent_tokenize(raw_query.strip().lower(), "english") # a sentence list from doc 
            if sent_tokenize_list: # if sent_tokenize_list is not empty
                for sent in sent_tokenize_list:
                    words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                    words = [word.strip(string.punctuation) for word in words]
                    clean_query+=words
            clean_query = filter(None, clean_query)
            clean_query_str = " ".join(clean_query)

            ### find any matched token
            valid_tokens = set(clean_query)&set(clean_token_index.keys())
            for token in valid_tokens:
                if token in clean_query: ## a token found in query
                    for term in clean_token_index.get(token):
                        if term in clean_query_str:
                            match_dict[term]=[clean_entry_term_mesh_dict.get(term)]
                            match_dict[term].append(clean_mesh_dict.get(term))
                            match_dict[term]=filter(None, match_dict[term])
            self.pmid_matched_mesh_dict[pmid] = [item for sublist in match_dict.values() for item in sublist]

        ## setting 2: lower case, remove puncutations, normalize words, stem words
        # for pmid, raw_query in self.query_dict.iteritems():
        #     match_dict = defaultdict() ## match terms

        #     ### clean this query
        #     clean_query = []
        #     sent_tokenize_list = sent_tokenize(raw_query.strip().lower(), "english") # a sentence list from doc 
        #     if sent_tokenize_list: # if sent_tokenize_list is not empty
        #         for sent in sent_tokenize_list:
        #             words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
        #             words = [word.strip(string.punctuation) for word in words]
        #             clean_query+=words
        #     clean_query = filter(None, clean_query)
        #     clean_query_str = " ".join(clean_query)

        #     ### find any matched token
        #     valid_tokens = set(clean_query)&set(clean_token_index.keys())
        #     for token in valid_tokens:
        #         if token in clean_query: ## a token found in query
        #             for term in clean_token_index.get(token):
        #                 if term in clean_query_str:
        #                     match_dict[term]=[clean_entry_term_mesh_dict.get(term)]
        #                     match_dict[term].append(clean_mesh_dict.get(term))
        #                     match_dict[term]=filter(None, match_dict[term])
        #     self.pmid_matched_mesh_dict[pmid] = [item for sublist in match_dict.values() for item in sublist]        

    def comp_query_text_coverage(self):
        '''compute the coverage of MeSH from queries'''
        mesh_num_list = []
        for q_pmid in self.query_dict.keys():
            ## load gold standard MeSH of query pmids from NLM2007
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load mesh candidates from the query text
            query_mesh = filter(None, self.pmid_matched_mesh_dict.get(q_pmid))
            mesh_num_list.append(len(query_mesh))
            ## compute coverage
            overlap = set(gstd)&set(query_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.query_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "The coverage of MeSH from query texts of NLM2007: ", sum(self.query_coverage_list)*1.0/len(self.query_coverage_list), len(self.query_coverage_list)        
        print "Average number of MeSH from query: ", sum(mesh_num_list)*1.0/len(mesh_num_list)

    def load_candidates_pmids_from_KNN(self):
        '''load candidates identified from KNN methods'''
        try:
            nlm2007_knn_pmids_from_1997_corpus_dict = cPickle.load(file(self.nlm2007_knn_pmid_from_1997_corpus_file))
        except:
            print "Run corpus_medline_mesh_coverage.py to get NLM2007_as_query_knn_from_%s.pkl"%self.period

        # return nlm2007_knn_pmids_from_1997_corpus_dict
        candidate_count_list = []
        for q_pmid, knn_pmid_score_pairs in nlm2007_knn_pmids_from_1997_corpus_dict.iteritems():
            # print q_pmid
            # print knn_pmid_score_pairs[:5]
            # raw_input('wait...')
            # knn_pmids = [x[0] for x in knn_pmid_score_pairs]
            knn_pmids = knn_pmid_score_pairs
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            self.std_mesh_dict[q_pmid] = gstd
            knn_mesh_list = []
            ## exclude q_pmid from knn_pmids
            knn_pmids = list(set(knn_pmids)-set([q_pmid]))
            for k_pmid in knn_pmids:
                k_mesh = file(os.path.join(self.mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                knn_mesh_list+=k_mesh
            knn_mesh = filter(None, list(set(knn_mesh_list)))
            self.knn_mesh_dict[q_pmid]=knn_mesh
            candidate_count_list.append(len(knn_mesh))
            ## coverage
            overlap = set(gstd)&set(knn_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.knn_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "Coverage of MeSH candidates from KNN papers %s corpus: "%self.period, sum(self.knn_coverage_list)*1.0/len(self.knn_coverage_list), len(self.knn_coverage_list)
        print "Average number of candidates from KNN papers %s corpus: "%self.period, sum(candidate_count_list)*1.0/len(candidate_count_list)

    def comp_coverage_joint_candidates(self):
        '''combine candidates from KNN and query texts, and compute the coverage'''
        cand_count_list = []
        q_pmid_query_mesh = defaultdict()
        for q_pmid in self.query_dict.keys():
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            query_mesh = filter(None, self.pmid_matched_mesh_dict.get(q_pmid))
            q_pmid_query_mesh[q_pmid] = list(set(query_mesh))
            knn_mesh = self.knn_mesh_dict.get(q_pmid)
            joint_mesh = set(query_mesh)|set(knn_mesh)
            cand_count_list.append(len(joint_mesh))
            ## coverage
            overlap = set(gstd)&joint_mesh
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.joint_coverage_list.append(coverage)
            except Exception as e:
                pass
        cPickle.dump(q_pmid_query_mesh, file(self.nlm2007_qpmid_mesh_from_query_file,"w"))
        print "Coverage of combined MeSH candidates from the query and KNN papers: ", sum(self.joint_coverage_list)*1.0/len(self.joint_coverage_list), len(self.joint_coverage_list)
        print "Averge number of joint candidates: ", sum(cand_count_list)*1.0/len(cand_count_list)

    def load_prc_candidates(self):
        '''load mesh from prc determined knn papers'''
        raw_data_dir = os.path.join(data_dir,"lu_data")
        clean_dir = os.path.join(data_dir, "lu_data", "clean")

        NLM2007 = Data(raw_data_dir, clean_dir)
        NLM2007.nlm2007()
        NLM2007_pmids = NLM2007.query_pmids
        NLM2007_prc_knns = NLM2007.nbr_dict

        cand_num_list = []
        prc_coverage_list = []
        for q_pmid in NLM2007_prc_knns.keys():
            knn_pmid_mesh_list = []
            knn_pmids = NLM2007_prc_knns.get(q_pmid)
            knn_pmids = [x[0] for x in knn_pmids]
            for k_pmid in knn_pmids:
                if len(NLM2007.nbr_tam[k_pmid])==3:
                    knn_pmid_mesh = self.mesh_parser4Lu_data(NLM2007.nbr_tam[k_pmid][-1])
                    knn_pmid_mesh_list+=knn_pmid_mesh
                # self.lu_pmid_mesh_dict[k_pmid] = knn_pmid_mesh
            knn_pmid_mesh_list = list(set(knn_pmid_mesh_list))
            cand_num_list.append(len(knn_pmid_mesh_list))
            ## coverag:
            gstd = self.std_mesh_dict.get(q_pmid)
            gstd = [x.lower() for x in gstd]
            overlap = set(gstd)&set(knn_pmid_mesh_list)
            coverage  = len(overlap)*1.0/len(gstd)
            prc_coverage_list.append(coverage)
        print "Average number of MeSH candidates from PRC KNN: ", sum(cand_num_list)*1.0/len(cand_num_list)
        print "Average coverage of MeSH from PRC KNN: ", sum(prc_coverage_list)*1.0/len(prc_coverage_list)

    def mesh_parser4Lu_data(self, text):
        text = re.sub("[-*&]"," ",text)
        mhList = text.split("!")
        mhList = [mh.strip("*") for mh in mhList]
        return mhList

class Lu_query_on_medline_1995_1997(Lu_query_on_medline1997):
    def __init__(self, data_dir, resource_dir):
        super(Lu_query_on_medline_1995_1997,self).__init__(data_dir, resource_dir)
        self.period = "1995_1997"
        self.mesh_1997_dir = os.path.join(self.analyDir, "mesh_%s"%self.period) ## from corpus_medline_mesh_coverage.py
        self.nlm2007_knn_pmid_from_1997_corpus_file = os.path.join(self.analyDir, "NLM2007_as_query_knn_from_%s.pkl"%self.period) ## from corpus_medline_mesh_coverage.py        
        self.nlm2007_qpmid_mesh_from_query_file = os.path.join(self.analyDir,"NLM2007_q_pmid_mesh_from_query_%s.pkl"%self.period)

    def run(self):
        self.load_query() ## {pmid:title+abstract text in raw format}
        self.screen_query()
        self.comp_query_text_coverage()
        self.load_candidates_pmids_from_KNN()
        self.load_prc_candidates()
        self.comp_coverage_joint_candidates()
        self.comp_candidate_both_in_query_and_knn()

    def comp_candidate_both_in_query_and_knn(self):
        '''Evaluate the role of candidates appearing in both the query and KNN papers'''
        prec_list = []
        recall_list = []
        for q_pmid in self.query_dict.keys():
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            query_mesh = filter(None, self.pmid_matched_mesh_dict.get(q_pmid))
            knn_mesh = self.knn_mesh_dict.get(q_pmid)
            
            double_match_mesh = set(query_mesh)&set(knn_mesh)

            ## coverage
            overlap = set(gstd)&double_match_mesh
            prec = len(overlap)*1.0/len(gstd)
            recall = len(overlap)*1.0/len(double_match_mesh)
            prec_list.append(prec)
            recall_list.append(recall)
        print "Average precision: ", sum(prec_list)*1.0/len(prec_list)
        print "Average recall: ", sum(recall_list)*1.0/len(recall_list)

class Lu_query_metamap_on_medline_1995_1997(Lu_query_on_medline1997):
    '''Use NLM2007 as query, extract candidates from queries using metamap'''
    def __init__(self, data_dir, resource_dir):
        super(Lu_query_metamap_on_medline_1995_1997,self).__init__(data_dir, resource_dir)
        self.mmDir = os.path.join(self.dataDir, "metamap_data") ## dir for metamap input files
        self.nlm2007_metamap_dir = os.path.join(self.mmDir, "nlm2007")
        self.nlm2007_metamap_in_dir = os.path.join(self.nlm2007_metamap_dir, "input")
        self.nlm2007_metamap_out_dir = os.path.join(self.nlm2007_metamap_dir, "output")
        self.metamap_out_file = os.path.join(self.nlm2007_metamap_dir, "nlm2007_pmid_cand_score_dict.pkl")
        self.metamap_out = defaultdict()

    def run(self):
        self.load_query() ## {pmid:title+abstract text in raw format}
        # self.call_metamap()
        self.analyze_metamap_result()
        self.comp_metamap_query_text_coverage()

    def call_metamap(self):
        '''call metamap to extract MeSH candidates'''
        ## generate files for metamap on NLM2007
        ## to call metamap, check corpus_medline_call_metamap.py
        nlm2007_metamap_dir = os.path.join(self.mmDir, "nlm2007")
        nlm2007_metamap_in_dir = os.path.join(nlm2007_metamap_dir, "input")
        nlm2007_metamap_out_dir = os.path.join(nlm2007_metamap_dir, "output")
        if not os.path.exists(nlm2007_metamap_dir):
            os.makedirs(nlm2007_metamap_dir)
        if not os.path.exists(nlm2007_metamap_in_dir):
            os.makedirs(nlm2007_metamap_in_dir)
        if not os.path.exists(nlm2007_metamap_out_dir):
            os.makedirs(nlm2007_metamap_out_dir)
        for pmid, text in self.query_dict.iteritems():
            fout = file(os.path.join(nlm2007_metamap_in_dir, "%s.in"%pmid),"w")
            fout.write(text+"\n\n")
    
    def analyze_metamap_result(self):
        '''extract candidates and scores from metamap prediction'''
        try:
            self.metamap_out = cPickle.load(file(self.metamap_out_file))
        except:
            for doc in os.listdir(self.nlm2007_metamap_out_dir):
                pmid=doc.split(".out")[0]
                self.metamap_out[pmid] = []
                fin = file(os.path.join(self.nlm2007_metamap_out_dir,doc))
                for line in fin:
                    rec = line.split("|")
                    if rec[1]=="MMI":
                        term = rec[3]
                        score = rec[2]
                        self.metamap_out[pmid].append((term, score))
            cPickle.dump(self.metamap_out, file(self.metamap_out_file,"w"))

    def comp_metamap_query_text_coverage(self):
        '''compute the coverage of MeSH from metamap'''
        mesh_num_list = []
        query_coverage_list = []
        for q_pmid in self.query_dict.keys():
            ## load gold standard MeSH of query pmids from NLM2007
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load mesh candidates from the query text
            mm_query_mesh = filter(None, self.metamap_out.get(q_pmid))
            mesh_num_list.append(len(mm_query_mesh))
            ## all lower case
            gstd = [x.lower() for x in gstd]
            mm_query_mesh = [x[0].lower() for x in mm_query_mesh]
            ## compute coverage
            overlap = set(gstd)&set(mm_query_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                query_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "The coverage of MeSH from query texts of NLM2007: ", sum(query_coverage_list)*1.0/len(query_coverage_list), len(query_coverage_list)        
        print "Average number of MeSH from query: ", sum(mesh_num_list)*1.0/len(mesh_num_list)

class Analysis_mesh_from_query(Lu_query_metamap_on_medline_1995_1997):
    '''Compare the performance of candidates from string matching and metamap. '''
    def __init__(self, data_dir, resource_dir):
        super(Analysis_mesh_from_query, self).__init__(data_dir, resource_dir)
        self.nlm2007_str_match_qpmid_mesh_dict = defaultdict()
        self.nlm2007_metamap_qpmid_mesh_dict = defaultdict()

    def run(self):     
        
        self.load_query() ## {pmid:title+abstract text in raw format}
        self.screen_query()
        self.comp_query_text_coverage()
        self.load_candidates_pmids_from_KNN() ## to get self.knn_mesh {k_pmid:[mesh cands]}
        self.load_prc_candidates()
        self.comp_coverage_joint_candidates()

        self.nlm2007_str_match_qpmid_mesh_dict = cPickle.load(file(self.nlm2007_qpmid_mesh_from_query_file))
        self.nlm2007_metamap_qpmid_mesh_dict = cPickle.load(file(self.metamap_out_file))

        self.comp_overlap_string_match_and_metamap() 
        self.combine_cand_from_query_and_knn()

    def comp_overlap_string_match_and_metamap(self):
        '''compare candidates from string matching and metamap'''
        str_mt_base_list = []
        metamap_base_list = []
        pmidList = self.nlm2007_metamap_qpmid_mesh_dict.keys()
        str_mt_prec_list = []
        str_mt_reca_list = []
        mm_prec_list = []
        mm_reca_list = []
        overlap_prec_list = []
        overlap_reca_list = []
        for q_pmid in pmidList:
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            gstd = [x.lower() for x in gstd]
            str_mt = self.nlm2007_str_match_qpmid_mesh_dict.get(q_pmid)
            str_mt = [x.lower() for x in str_mt]
            mm_mt = self.nlm2007_metamap_qpmid_mesh_dict.get(q_pmid)
            mm_mt = [x[0].lower() for x in mm_mt]
            overlap = set(str_mt)&set(mm_mt)
            str_mt_base_list.append(len(overlap)*1.0/len(str_mt))
            metamap_base_list.append(len(overlap)*1.0/len(mm_mt))
            str_mt_prec_list.append(len(set(gstd)&set(str_mt))*1.0/len(gstd))
            mm_prec_list.append(len(set(gstd)&set(mm_mt))*1.0/len(gstd))
            str_mt_reca_list.append(len(set(gstd)&set(str_mt))*1.0/len(str_mt))
            mm_reca_list.append(len(set(gstd)&set(mm_mt))*1.0/len(mm_mt))
            overlap_prec_list.append(len(set(overlap)&set(gstd))*1.0/len(gstd))
            overlap_reca_list.append(len(set(overlap)&set(gstd))*1.0/len(overlap))

        print "String matching vs. Metamap: ", sum(str_mt_base_list)*1.0/len(str_mt_base_list)
        print "MetaMap vs. String matching: ", sum(metamap_base_list)*1.0/len(metamap_base_list)
        print "String matching average prec: ", sum(str_mt_prec_list)*1.0/len(str_mt_prec_list)
        print "MetaMap average precision: ", sum(mm_prec_list)*1.0/len(mm_prec_list)
        print "String matching average recall: ", sum(str_mt_reca_list)*1.0/len(str_mt_reca_list)
        print "MetaMap average recall: ", sum(mm_reca_list)*1.0/len(mm_reca_list)
        print "Overlap average precision: ", sum(overlap_prec_list)*1.0/len(overlap_prec_list)
        print "Overlap average recall: ", sum(overlap_reca_list)*1.0/len(overlap_reca_list)

    def combine_cand_from_query_and_knn(self):
        '''combine candidates from query (using either string match or metamap) and BM25 KNN '''  
        print "self.nlm2007_str_match_qpmid_mesh_dict ", len(self.nlm2007_str_match_qpmid_mesh_dict)
        print "self.nlm2007_metamap_qpmid_mesh_dict ", len(self.nlm2007_metamap_qpmid_mesh_dict)
        print
        count_str_mt_and_knn = []
        count_mm_and_knn = []
        count_str_mt_and_mm_and_knn = []
        count_overlap_and_knn = []
        count_overlap_and_str_mt_and_knn = []
        count_overlap_and_mm_and_knn = []

        joint_str_mt_and_knn = []
        joint_mm_and_knn = []
        joint_str_mt_and_mm_and_knn = []
        joint_overlap_and_knn = []
        joint_overlap_and_str_mt_and_knn = []
        joint_overlap_and_mm_and_knn = []

        q_pmid_query_mesh = defaultdict()
        q_pmid_list = self.nlm2007_str_match_qpmid_mesh_dict.keys()
        for q_pmid in q_pmid_list:
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            str_mt_query_mesh = filter(None, self.nlm2007_str_match_qpmid_mesh_dict.get(q_pmid)) ## self.pmid_matched_mesh_dict==self.nlm2007_str_match_qpmid_mesh_dict
            mm_query_mesh = filter(None, self.nlm2007_metamap_qpmid_mesh_dict.get(q_pmid))
            overlap = set(str_mt_query_mesh)&set(mm_query_mesh)

            knn_mesh = self.knn_mesh_dict.get(q_pmid)

            ## various combinations
            str_mt_and_knn = set(str_mt_query_mesh)|set(knn_mesh)
            mm_and_knn = set(mm_query_mesh)|set(knn_mesh)
            str_mt_and_mm_and_knn = set(str_mt_query_mesh)|set(mm_query_mesh)|set(knn_mesh)
            overlap_and_knn = overlap|set(knn_mesh)
            str_mt_and_overlap_and_knn = set(str_mt_query_mesh)|overlap|set(knn_mesh)
            mm_and_overlap_and_knn = set(mm_query_mesh)|overlap|set(knn_mesh)

            ## candidate set size
            count_str_mt_and_knn.append(len(str_mt_and_knn))
            count_mm_and_knn.append(len(mm_and_knn))
            count_str_mt_and_mm_and_knn.append(len(str_mt_and_mm_and_knn))
            count_overlap_and_knn.append(len(overlap_and_knn))
            count_overlap_and_str_mt_and_knn.append(len(str_mt_and_overlap_and_knn))
            count_overlap_and_mm_and_knn.append(len(mm_and_overlap_and_knn))

            ## coverage
            overlap_str_mt_and_knn = set(gstd)&str_mt_and_knn
            overlap_mm_and_knn = set(gstd)&mm_and_knn
            overlap_str_mt_and_mm_and_knn = set(gstd)&str_mt_and_mm_and_knn
            overlap_overlap_and_knn = set(gstd)&overlap_and_knn
            overlap_str_mt_and_overlap_and_knn = set(gstd)&str_mt_and_overlap_and_knn
            overlap_mm_and_overlap_and_knn = set(gstd)&mm_and_overlap_and_knn
            try:
                coverage_str_mt_and_knn = len(overlap_str_mt_and_knn)*1.0/len(gstd)
                joint_str_mt_and_knn.append(coverage_str_mt_and_knn)
            except Exception as e:
                print "coverage_str_mt_and_knn"
                print e
                raw_input('...')

            try:
                coverage_mm_and_knn = len(overlap_mm_and_knn)*1.0/len(gstd)
                joint_mm_and_knn.append(coverage_mm_and_knn)
            except Exception as e:
                print "coverage_mm_and_knn"
                print e
                raw_input('...')

            try:
                coverage_str_mt_and_mm_and_knn = len(overlap_str_mt_and_mm_and_knn)*1.0/len(gstd)
                joint_str_mt_and_mm_and_knn.append(coverage_str_mt_and_mm_and_knn)
            except Exception as e:
                print "coverage_str_mt_and_mm_and_knn"
                print e
                raw_input('...')               

            try:
                coverage_overlap_and_knn = len(overlap_overlap_and_knn)*1.0/len(gstd)
                joint_overlap_and_knn.append(coverage_overlap_and_knn)
            except Exception as e:
                print "coverage_overlap_and_knn"
                print e
                raw_input('...') 
            try:
                coverage_str_mt_and_overlap_and_knn = len(overlap_str_mt_and_overlap_and_knn)*1.0/len(gstd)
                joint_overlap_and_str_mt_and_knn.append(coverage_str_mt_and_overlap_and_knn)
            except Exception as e:
                print "coverage_str_mt_and_overlap_and_knn"
                print e
                raw_input('...') 

            try:
                coverage_mm_and_overlap_and_knn = len(overlap_mm_and_overlap_and_knn)*1.0/len(gstd)
                joint_overlap_and_mm_and_knn.append(coverage_mm_and_overlap_and_knn)
            except Exception as e:
                print "coverage_mm_and_overlap_and_knn"
                print e
                raw_input('...')

        print "====================="
        print "Coverage of combined candidates from query and KNN papers"
        print "Joint set coverage summary\n"
        print "String match and BM25 KNN: ", sum(joint_str_mt_and_knn)*1.0/len(joint_str_mt_and_knn), len(joint_str_mt_and_knn)
        print "Average number of joint candidates: ", sum(count_str_mt_and_knn)*1.0/len(count_str_mt_and_knn)
        print
        print "MetaMap and BM25 KNN: ", sum(joint_mm_and_knn)*1.0/len(joint_mm_and_knn), len(joint_mm_and_knn)
        print "Average number of joint candidates: ", sum(count_mm_and_knn)*1.0/len(count_mm_and_knn)
        print 
        print "String match, MetaMap and BM25 KNN: ", sum(joint_str_mt_and_mm_and_knn)*1.0/len(joint_str_mt_and_mm_and_knn), len(joint_str_mt_and_mm_and_knn)
        print "Average number of joint candidates: ", sum(count_str_mt_and_mm_and_knn)*1.0/len(count_str_mt_and_mm_and_knn)
        print 
        print "Overlap and BM25 KNN: ", sum(joint_overlap_and_knn)*1.0/len(joint_overlap_and_knn), len(joint_overlap_and_knn)
        print "Average number of joint candidates: ", sum(count_overlap_and_knn)*1.0/len(count_overlap_and_knn)
        print 
        print "Overlap, string match, and BM25 KNN: ", sum(joint_overlap_and_str_mt_and_knn)*1.0/len(joint_overlap_and_str_mt_and_knn), len(joint_overlap_and_str_mt_and_knn)
        print "Average number of joint candidates: ", sum(count_overlap_and_str_mt_and_knn)*1.0/len(count_overlap_and_str_mt_and_knn)
        print
        print "Overlap, MetaMap, and BM25 KNN: ", sum(joint_overlap_and_mm_and_knn)*1.0/len(joint_overlap_and_mm_and_knn), len(joint_overlap_and_mm_and_knn)
        print "Average number of joint candidates: ", sum(count_overlap_and_mm_and_knn)*1.0/len(count_overlap_and_mm_and_knn)
        print               
 

if __name__=="__main__":
    data_dir = "/home/w2wei/data"
    nlm_dir = os.path.join(data_dir, "nlm_data")
    mesh2016_file = os.path.join(nlm_dir, "d2016.bin")
    ## entry_term - mesh dict in raw format
    entry_term_mesh_dict_file = os.path.join(nlm_dir, "entry_term_mesh_dict.pkl")
    build_entry_term_mesh_dict(mesh2016_file, entry_term_mesh_dict_file)
    ## mesh term - mesh dict in raw format
    mesh_dict_file = os.path.join(nlm_dir, "mesh_vocab.pkl")
    build_mesh_vocab(mesh2016_file, mesh_dict_file)
    ## cleaned entry_term - mesh dict
    clean_entry_term_mesh_dict_file = os.path.join(nlm_dir, "clean_entry_term_mesh_dict.pkl")
    build_clean_entry_term_mesh_dict(entry_term_mesh_dict_file, clean_entry_term_mesh_dict_file)
    ## cleaned mesh - mesh dict
    clean_mesh_dict_file = os.path.join(nlm_dir, "clean_mesh_dict.pkl")
    build_clean_mesh_vocab(mesh_dict_file, clean_mesh_dict_file)
    ## index n-grams in mesh terms and entry terms
    mesh_and_entry_token_index_file = os.path.join(nlm_dir, "mesh_and_entry_terms_token_index.pkl")
    index_ngrams(clean_entry_term_mesh_dict_file, clean_mesh_dict_file, mesh_and_entry_token_index_file)


    ## Experiment 1, on MEDLINE 1997 corpus
    exp1 = Lu_query_on_medline1997(data_dir, nlm_dir)
    # exp1.run()
    ## Experiment 2, on MEDLINE 1995-1997 corpus
    exp2 = Lu_query_on_medline_1995_1997(data_dir, nlm_dir)
    # exp2.run()
    ## Experiment 3, use MetaMap to extract candidates from NLM2007
    exp3 = Lu_query_metamap_on_medline_1995_1997(data_dir, nlm_dir)
    # exp3.run()
    ## Experiment 4, compare candidates from string matching and metamap
    exp4 = Analysis_mesh_from_query(data_dir, nlm_dir)
    # exp4.run()