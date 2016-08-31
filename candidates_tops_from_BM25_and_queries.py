'''
    Get top candidates from BM25 KNN and queries

    Created on July 29, 2016
    Updated on August 30, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from collections import defaultdict
from knn_data import Data
from collections import Counter
from operator import itemgetter
from candidates_from_BM25KNN import BM25KNN_candidates

class Top_candidates_from_BM25_and_queries(BM25KNN_candidates):
    '''Evaluate the average coverage of combined top candidates from both BM25KNN papers and queries '''
    def __init__(self, data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file, top_k):
        super(Eval_top_candidates_from_BM25_and_queries, self).__init__(data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file)
        self.mesh_from_query_file = os.path.join(self.out_dir,"%s_qpmid_mesh_candidates_from_queries.pkl"%(self.query_dataset))
        self.top_cands_file = os.path.join(self.out_dir, "%s_candidates_from_queries_and_top_%d_BM25KNN_on_MEDLINE_%s.pkl"%(self.query_dataset, top_k, self.period))
        self.query_std_mesh_dict = defaultdict()
        self.qpmid_matched_mesh_dict = defaultdict()
        self.top_cands_dict = defaultdict()
        self.top_k = top_k

    def run(self):
        self.load_top_BM25KNN_candidates()
        self.load_candidates_from_queries()
        self.get_top_candidates()

    def load_top_BM25KNN_candidates(self):
        '''load top candidates from BM25KNN'''
        try:
            self.top_cand_dict = cPickle.load(file(self.qpmid_top_cand_file))
        except:
            print "Top candidates from BM25KNN are not yet ready. Run candidates_find_tops_BM25KNN.py"
            sys.exit(1)

    def load_candidates_from_queries(self):
        '''Load candidates extracted from query texts using string matching'''
        try:
            self.qpmid_matched_mesh_dict = cPickle.load(file(self.mesh_from_query_file))
        except:
            print "Error: Extracted MeSH candidates from queries are not yet ready! Run candidates_extract_from_query.py"
            sys.exit(1)

    def get_top_candidates(self):
        '''Get top candidates from BM25KNN and queries'''
        try:
            self.top_cands_dict = cPickle.load(file(self.top_cands_file))
        except:
            query_pmid_list = self.query_std_mesh_dict.keys()
            for q_pmid in query_pmid_list:
                top_knn_cands = self.top_cand_dict.get(q_pmid)[:self.top_k]
                cands_from_query = self.qpmid_matched_mesh_dict.get(q_pmid)
                joint_set = set(top_knn_cands)|set(cands_from_query)
                self.top_cands_dict[q_pmid] = list(joint_set)
            cPickle.dump(self.top_cands_dict, file(self.top_cands_file,"wb"))
        print "%s top candidates ready"%(self.query_dataset)

if __name__=="__main__":
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)
    
    top_k = int(sys.argv[2])

    data_dir = '/home/w2wei/data'
    data2_dir = '/home/w2wei/data2'
    analysis_dir = os.path.join(data2_dir, 'analysis', 'L1000_query_PMID_on_MEDLINE_%s'%time_span)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    util_dir = os.path.join(data2_dir, "utils")
    pmid_year_dict_file = os.path.join(util_dir, "pmid_year_dict.pkl")

    knn_base_dir = os.path.join(data2_dir, "knn")

    L1000_top_cands = Top_candidates_from_BM25_and_queries(data_dir, data2_dir, knn_base_dir, analysis_dir, time_span, pmid_year_dict_file, top_k)
    L1000_top_cands.run()