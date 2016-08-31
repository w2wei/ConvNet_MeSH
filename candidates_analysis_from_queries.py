'''
    Analyze the coverage/AP of MeSH candidates from queries using string matching.

    Created on August 26, 2016
    Updated on August 26, 2016
    @author: Wei Wei
'''

import os, cPickle, string, time, re, sys
from collections import defaultdict
from knn_data import Data
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
import multiprocessing as mp
from candidates_extract_from_query import *

class Eval_candidates_from_queries(Candidates_from_queries):
    '''Evaluate the coverage of candidates extracted from queries using string matching'''  

    def __init__(self, data_dir, data2_dir, resource_dir, analysis_dir, time_span, data_name, pmid_year_dict_file, mesh_base_dir):
        super(Eval_candidates_from_queries, self).__init__(data_dir, data2_dir, resource_dir, analysis_dir, time_span, data_name)
        self.pmid_year_dict = cPickle.load(file(pmid_year_dict_file))
        self.mesh_base_dir = mesh_base_dir
        self.query_dict = defaultdict() ## {pmid:title+abstract text in raw format}
        self.std_mesh_dict = defaultdict() 
        self.knn_mesh_dict = defaultdict() ## {query pmid: knn mesh}
        self.query_coverage_list = [] ## list for the coverage of query text, i.e., mesh terms recognized from query texts

    def run(self):
        self.load_query()
        self.load_extracted_mesh()
        self.comp_query_text_coverage()

    def load_extracted_mesh(self):
        '''screen queries and identify mesh candidates'''
        try:
            self.pmid_matched_mesh_dict = cPickle.load(file(self.mesh_from_query_file))
        except:
            print "Error: Extracted MeSH from queries not yet ready! Run candidates_extract_from_query.py"
            sys.exit(1)

    def comp_query_text_coverage(self):
        '''compute the coverage of MeSH from queries'''
        mesh_num_list = []
        print "totoal query pmid num: ", len(self.query_dict)
        miss_gstd_num = 0
        for q_pmid in self.query_dict.keys():
            ## load gold standard MeSH of query pmids from NLM2007
            try:
                gstd_year = self.pmid_year_dict.get(q_pmid)
                gstd_mesh_file = os.path.join(self.mesh_base_dir,gstd_year,"%s.txt"%q_pmid)
                gstd = filter(None, file(gstd_mesh_file).read().split('\n'))
            except Exception as e:
                print q_pmid
                print e
                # print "Missing MeSH gold standard: ", q_pmid
                miss_gstd_num+=1
                continue
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
        print "Missing gold std num: ", miss_gstd_num
        print "Average precision of MeSH from %s query on MEDLINE %s: "%(self.dataName, self.period), sum(self.query_coverage_list)*1.0/len(self.query_coverage_list), len(self.query_coverage_list)        
        # Average precision of MeSH from L1000 query on MEDLINE 2006_2009:  0.318500589656 1000
        print "Average number of MeSH from query: ", sum(mesh_num_list)*1.0/len(mesh_num_list)
        # Average number of MeSH from query:  18.684 

if __name__=="__main__":
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)

    data_dir = "/home/w2wei/data"
    nlm_dir = os.path.join(data_dir, "nlm_data")
    data2_dir = "/home/w2wei/data2"
    util_dir = os.path.join(data2_dir, "utils")
    mesh_base_dir = os.path.join(data2_dir, "mesh_by_year")
    pmid_year_dict_file = os.path.join(util_dir, "pmid_year_dict.pkl")
    mesh2016_file = os.path.join(nlm_dir, "d2016.bin")
    data_name = "L1000"
    
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

    exp = Eval_candidates_from_queries(data_dir, data2_dir, nlm_dir, analysis_dir, time_span, data_name, pmid_year_dict_file, mesh_base_dir)
    exp.run()
