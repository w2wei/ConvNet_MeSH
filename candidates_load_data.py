'''
    Prepare sample PMIDs

    Created on August 31, 2016
    Updated on August 31, 2016
    @author: Wei Wei
'''

import os, sys, random, cPickle, time
import knn_data as kdata
from collections import defaultdict
from candidates_retrieve import Retrieve, submit_queries

class Data(object):
    '''The data class for any pmid'''
    def __init__(self, data2_dir, exp_dir, index_dir, time_span, sample_size, pmid_year_dict_file, rm_list):        

        self.data2_dir = data2_dir
        self.exp_dir = exp_dir
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)
        self.index_dir = index_dir
        self.index_file = "medline_%s.index"%(time_span)
        startyear, endyear = time_span.split("_")
        self.years = range(int(startyear), int(endyear)+1)
        self.sample_size = sample_size
        self.pmid_year_dict = cPickle.load(file(pmid_year_dict_file))
        self.rm_list = rm_list

        self.pmid_base_dir = os.path.join(self.data2_dir, "pmid_docs_by_year")
        self.tiab_base_dir = os.path.join(self.data2_dir, "tiab_by_year")
        self.mesh_base_dir = os.path.join(self.data2_dir, "mesh_by_year")

        self.query_pmids_file = os.path.join(self.exp_dir, "%s_query.pmids"%self.sample_size)
        self.query_tam_file = os.path.join(self.exp_dir, "%s_query.tam"%self.sample_size)
        self.nbr_dict_file = os.path.join(self.exp_dir, "%s_nbr.pmids"%self.sample_size)
        self.nbr_tam_file = os.path.join(self.exp_dir, "%s_nbr.tam"%self.sample_size)

        ## attributes of a data instance
        self.query_pmids = []
        self.query_tam = defaultdict()
        self.nbr_tam = defaultdict()
        self.nbr_dict = defaultdict()

    def run(self):
        self.get_query_pmids()
        self.get_query_tam()
        self.get_nbr_dict()
        self.get_nbr_tam()

    def get_query_pmids(self):
        try:
            self.query_pmids = cPickle.load(file(self.query_pmids_file))
        except:
            pmidList = []
            pmid_dirs = [os.path.join(self.pmid_base_dir, str(year)) for year in self.years]
            for pmid_dir in pmid_dirs:
                for fname in os.listdir(pmid_dir):
                    pmids = filter(None, file(os.path.join(pmid_dir, fname)).read().split("\n"))
                    pmidList+=pmids
            pmidList = list(set(pmidList)-set(self.rm_list))
            random.shuffle(pmidList)
            self.query_pmids = random.sample(pmidList, int(self.sample_size))
            cPickle.dump(self.query_pmids, file(self.query_pmids_file, "w"))
        print "Query pmid num %d, given sample size %s"%(len(self.query_pmids),self.sample_size)

    def get_query_tam(self):
        '''Get the title, the abstract and the MeSH terms for every sample PMID'''
        try:
            self.query_tam = cPickle.load(file(self.query_tam_file))
        except:
            valid_q_pmid = []
            for q_pmid in self.query_pmids:
                # try:
                q_pmid_year = self.pmid_year_dict.get(q_pmid)
                q_tiab_file = os.path.join(self.tiab_base_dir, q_pmid_year, "%s.txt"%q_pmid)
                q_tiab_raw = filter(None, file(q_tiab_file).read().split("\n")[:2])
                if len(q_tiab_raw)<2: ## if a pmid does not have a title or an abstract, skip it
                    continue
                q_mesh_file = os.path.join(self.mesh_base_dir, q_pmid_year, "%s.txt"%q_pmid)
                q_mesh_raw = filter(None, file(q_mesh_file).read().split("\n"))
                if not q_mesh_raw: ## if a pmid does not have mesh terms, skip it
                    continue
                ## if both raw tiab and mesh are qualified, clean them
                q_ttl, q_abs = q_tiab_raw
                q_mesh = "!".join([x.lower() for x in q_mesh_raw])
                self.query_tam[q_pmid] = [[q_ttl], [q_abs], q_mesh]
                valid_q_pmid.append(q_pmid)
            cPickle.dump(self.query_tam, file(self.query_tam_file, "w"))
            self.query_pmids = valid_q_pmid[:int(self.sample_size)] ## refine the query pmid list
            print "Adjusted query pmid num %d."%len(self.query_pmids)
            cPickle.dump(self.query_pmids, file(self.query_pmids_file, "w")) ## save the updated query pmids
        print "Query tam num %d. "%len(self.query_tam)

    def get_nbr_dict(self):
        '''Get BM25KNN pmids for every query pmid'''
        try:
            self.nbr_dict = cPickle.load(file(self.nbr_dict_file))
        except:
            query_dict = {}
            for pmid, tam in self.query_tam.iteritems():
                query_dict[pmid] = ". ".join([tam[0][0], tam[1][0]])
            print "Query size: ",len(query_dict)
            self.nbr_dict = submit_queries(query_dict, self.index_dir, self.index_file, self.nbr_dict_file)
        print "%s sample nbr dict ready"%self.sample_size

    def get_nbr_tam(self):
        '''Get titles, abstracts, and MeSH terms of neighbor PMIDs'''
        try:
            self.nbr_tam = cPickle.load(file(self.nbr_tam_file))
        except:
            nbr_pmids = self.nbr_dict.values()
            nbr_pmids = [item for sublist in nbr_pmids for item in sublist]
            nbr_pmids = [x[0] for x in nbr_pmids]
            valid_n_pmid = []
            for n_pmid in nbr_pmids:
                n_pmid_year = self.pmid_year_dict.get(n_pmid)
                n_tiab_file = os.path.join(self.tiab_base_dir, n_pmid_year, "%s.txt"%n_pmid)
                n_tiab_raw = filter(None, file(n_tiab_file).read().split("\n")[:2])
                if len(n_tiab_raw)<2: ## if a pmid does not have a title or an abstract, skip it
                    continue
                n_mesh_file = os.path.join(self.mesh_base_dir, n_pmid_year, "%s.txt"%n_pmid)
                n_mesh_raw = filter(None, file(n_mesh_file).read().split("\n"))
                if not n_mesh_raw: ## if a pmid does not have mesh terms, skip it
                    continue
                ## if both raw tiab and mesh are qualified, clean them
                n_ttl, n_abs = n_tiab_raw
                n_mesh = "!".join([x.lower() for x in n_mesh_raw])
                self.nbr_tam[n_pmid] = [[n_ttl], [n_abs], n_mesh]
                valid_n_pmid.append(n_pmid)
            cPickle.dump(self.nbr_tam, file(self.nbr_tam_file, "w"))
            print "total n pmid num: ", len(nbr_pmids)
            print "valid_n_pmid num: ", len(valid_n_pmid)
        print "nbr tam ready"
   
if __name__=="__main__":
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)

    sample_size = sys.argv[2]

    exp_dir_name = sys.argv[3]

    data_dir = "/home/w2wei/data"
    data2_dir = "/home/w2wei/data2"
    exp_dir = os.path.join(data_dir, "my_data",exp_dir_name)

    util_dir = os.path.join(data2_dir, "utils")
    pmid_year_dict_file = os.path.join(util_dir, "pmid_year_dict.pkl")
    index_dir = os.path.join(data2_dir,"index")

    testset_file = sys.argv[4] ## L1000, NLM2007, etc
    rm_list = filter(None, file(os.path.join(data_dir, "lu_data", testset_file, "%s.pmids"%testset_file)).read().split("\n")) ## rm_list is the test set
    print "rm_list from %s size: "%testset_file, len(rm_list)

    t0=time.time()
    sample = Data(data2_dir, exp_dir, index_dir, time_span, sample_size, pmid_year_dict_file, rm_list)
    sample.run()
    t1=time.time()
    print "time: ", t1-t0
