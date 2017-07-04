'''
    Prepare sample PMIDs

    Created on August 31, 2016
    Updated on August 31, 2016
    @author: Wei Wei
'''

import os, sys, random, cPickle, time
from collections import defaultdict

class Data(object):
    '''The data class for any pmid'''
    def __init__(self, exp_dir, sample_size,rm_list):        

        self.exp_dir = exp_dir
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        self.sample_size = sample_size
        self.rm_list = rm_list

        # self.query_pmids_file = os.path.join(self.exp_dir, "%s_query.pmids"%self.sample_size)
        self.query_tam_file = os.path.join(self.exp_dir, "%s_query.tam"%self.sample_size)
        self.nbr_dict_file = os.path.join(self.exp_dir, "%s_nbr.pmids"%self.sample_size)
        self.nbr_tam_file = os.path.join(self.exp_dir, "%s_nbr.tam"%self.sample_size)

        ## attributes of a data instance
        self.query_pmids = []
        self.query_tam = defaultdict()
        self.nbr_tam = defaultdict()
        self.nbr_dict = defaultdict()

    def run(self):
        self.get_query_tam()
        self.get_query_pmids()
        self.get_nbr_dict()
        self.get_nbr_tam()

    def get_query_pmids(self):
        try:
            # self.query_pmids = cPickle.load(file(self.query_pmids_file))
            self.query_pmids = self.query_tam.keys()
            self.query_pmids = list(set(self.query_pmids)-set(self.rm_list))
        except:
            print "run candidates_load_data.py"
        # print "%s sample PMIDs ready"%self.sample_size

    def get_query_tam(self):
        '''Get the title, the abstract and the MeSH terms for every sample PMID'''
        try:
            self.query_tam = cPickle.load(file(self.query_tam_file))
        except:
            print "run candidates_load_data.py"
        # print "Query pmid num %d. "%len(self.query_pmids)

    def get_nbr_dict(self):
        '''Get BM25KNN pmids for every query pmid'''
        try:
            self.nbr_dict = cPickle.load(file(self.nbr_dict_file))
        except:
            print "run candidates_load_data.py"
        # print "%s sample nbr dict ready"%self.sample_size

    def get_nbr_tam(self):
        '''Get titles, abstracts, and MeSH terms of neighbor PMIDs'''
        try:
            self.nbr_tam = cPickle.load(file(self.nbr_tam_file))
        except:
            print "run candidates_load_data.py"
        # print "nbr tam ready"
   
