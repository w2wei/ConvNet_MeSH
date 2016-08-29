'''
    Analyze the average precision of recent PMIDs' BM25 KNN candidates from the latest 1M MEDLINE

    Created on July 29, 2016
    Updated on August 8, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from candidates_retrieve import Retrieve
from collections import defaultdict

class Recent_pmids_as_query(object):
    '''Use pmids of late as queries. If most MeSH terms associated with these pmids are covered in KNN from latest 1M docs,\
       may need adjust reference corpus for every query.
    '''
    def __init__(self):
        self.data_dir = '/home/w2wei/data'
        self.query_file_dir = os.path.join(self.data_dir, 'latest_3M_tiab') ## queries from latest documents
        self.mesh_file_dir = os.path.join(self.data_dir, 'latest_3M_mesh')
        self.index_dir = os.path.join(self.data_dir, "index")
        self.index_file = "latest_3M.index"
        self.out_dir = os.path.join(self.data_dir, "latest_3M_analysis")
        self.knn_file = os.path.join(self.out_dir, "recent_docs_as_query_knn_from_latest_3M.pkl") ## result file

        self.query_dict = {}
        self.result = defaultdict()
        self.coverage_list = []

    def run(self):
        self.submit_queries()
        self.comp_coverage()

    def load_queries(self):
        '''prepare queries from latest 3M'''
        query_docs = random.sample(os.listdir(self.query_file_dir),500)## 200 random queries
        for q_doc in query_docs:
            pmid = q_doc.split(".txt")[0]
            text = " ".join(file(os.path.join(self.query_file_dir,q_doc)).read().split("\n")[:2])
            self.query_dict[pmid]=text

    def submit_queries(self):
        '''submit queries to index 3M documents'''
        try:
            self.result = cPickle.load(file(self.knn_file))
        except:
            self.load_queries()
            ret = Retrieve(self.index_dir, self.index_file, self.query_dict)
            self.result = ret.result
            cPickle.dump(self.result, file(self.knn_file, "w"))       

    def comp_coverage(self):
        for q_pmid, knn_pmids in self.result.iteritems():
            ## load gold standard MeSH of query pmids
            gstd = filter(None, file(os.path.join(self.mesh_file_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load mesh of knn pmids
            knn_mesh_list = []
            ## exclude q_pmid from knn_pmids
            knn_pmids = list(set(knn_pmids)-set([q_pmid]))
            for k_pmid in knn_pmids:
                k_mesh = file(os.path.join(self.mesh_file_dir,"%s.txt"%k_pmid)).read().split('\n')
                knn_mesh_list+=k_mesh
            knn_mesh = filter(None, list(set(knn_mesh_list)))
            ## compute coverage
            overlap = set(gstd)&set(knn_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.coverage_list.append(coverage)
            except Exception as e:
                pass
        print "Recent PMIDs' average coverage on latest 1M: ", sum(self.coverage_list)*1.0/len(self.coverage_list), len(self.coverage_list)

if __name__=="__main__":
    ## evaluate the coverage of recent pmids
    recent_pmids = Recent_pmids_as_query()
    recent_pmids.run()
