'''
    Extract MeSH candidates from queries using string matching.

    Created on August 26, 2016
    Updated on August 26, 2016
    @author: Wei Wei
'''

import os, cPickle, string, time, re, sys
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

class Candidates_from_queries(object):
    '''Find candidates from queries using string matching'''
    def __init__(self, data_dir, data2_dir, resource_dir, analysis_dir, time_span, data_name):
        self.dataName = data_name ##  L1000, NLM2007, etc
        self.dataDir = data_dir
        self.data2Dir = data2_dir
        self.dictDir = resource_dir ## resources from NLM, organized in dict format
        self.period = time_span ## "1997"
        self.analyDir = analysis_dir ## os.path.join(self.dataDir, "latest_3M_analysis") ## dir for mesh terms
        self.mesh_from_query_file = os.path.join(self.analyDir,"%s_qpmid_mesh_candidates_from_queries.pkl"%(self.dataName))
        self.pmid_matched_mesh_dict = defaultdict() ## matched mesh terms in every query
        self.query_dict = defaultdict()

    def run(self):
        self.load_query() ## {pmid:title+abstract text in raw format}
        self.screen_query()

    def load_query(self):
        '''load lu's data from knn_data'''
        raw_data_dir = os.path.join(self.dataDir,"lu_data")
        clean_dir = os.path.join(self.dataDir, "lu_data", "clean")

        data_obj = Data(raw_data_dir, clean_dir)
        if self.dataName=="L1000":
            data_obj.large1000()
        if self.dataName=="NLM2007":
            data_obj.nlm2007()

        data_obj_pmids = data_obj.query_pmids
        for pmid in data_obj_pmids:
            title, abstract, _ = data_obj.query_tam[pmid]
            self.query_dict[pmid] = ". ".join(title + abstract)

    def screen_query(self):
        '''screen queries and identify mesh candidates'''
        try:
            self.pmid_matched_mesh_dict = cPickle.load(file(self.mesh_from_query_file))
        except:
            ## load resources
            clean_token_index = cPickle.load(file(os.path.join(self.dictDir, "mesh_and_entry_terms_token_index.pkl"))) ## a dict, {token:[clean mesh or entry]}
            clean_entry_term_mesh_dict = cPickle.load(file(os.path.join(self.dictDir, "clean_entry_term_mesh_dict.pkl")))
            clean_mesh_dict = cPickle.load(file(os.path.join(self.dictDir, "clean_mesh_dict.pkl")))  
            ## setting: lower case, remove punctuation, normalize words
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
            cPickle.dump(self.pmid_matched_mesh_dict, file(self.mesh_from_query_file,"wb"), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    data_name = "L1000"
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)

    data_dir = "/home/w2wei/data"
    nlm_dir = os.path.join(data_dir, "nlm_data")
    mesh2016_file = os.path.join(nlm_dir, "d2016.bin")

    data2_dir = "/home/w2wei/data2"
    analysis_dir = os.path.join(data2_dir, 'analysis', '%s_query_PMID_on_MEDLINE_%s'%(data_name, time_span))
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

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

    ## Experiment
    exp = Candidates_from_queries(data_dir, data2_dir, nlm_dir, analysis_dir, time_span, data_name)
    exp.run()
