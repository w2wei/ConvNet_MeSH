'''
    Analyze the average precision of top candidates from BM25 KNN and queries

    Created on July 29, 2016
    Updated on August 30, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from collections import defaultdict
from knn_data import Data
from collections import Counter
from operator import itemgetter
from candidates_tops_from_BM25_and_queries import Top_candidates_from_BM25_and_queries

class Eval_top_candidates_from_BM25_and_queries(Top_candidates_from_BM25_and_queries):
    '''Evaluate the average coverage of combined top candidates from both BM25KNN papers and queries '''
    def __init__(self, data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file, top_k):
        super(Eval_top_candidates_from_BM25_and_queries, self).__init__(data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file)
        self.qpmid_std_mesh_file = os.path.join(self.out_dir, "%s_qpmid_std_mesh.pkl"%(self.query_dataset)) ## gold standard MeSH terms of query PMIDs

    def run(self):
        self.load_std_mesh()
        self.load_top_cands()
        self.eval_top_candidates()

    def load_std_mesh(self):
        '''Load gold standard MeSH terms of query PMIDs'''
        try:
            self.query_std_mesh_dict = cPickle.load(file(self.qpmid_std_mesh_file))
        except:
            miss_gstd_num=0
            for q_pmid, knn_pmid_score_pairs in self.bm25knn_result.iteritems():
                try:
                    gstd_year = self.pmid_year_dict.get(q_pmid)
                    gstd_mesh_file = os.path.join(self.mesh_base_dir,gstd_year,"%s.txt"%q_pmid)
                    gstd = filter(None, file(gstd_mesh_file).read().split('\n'))
                    self.query_std_mesh_dict[q_pmid]=gstd
                except Exception as e:
                    print q_pmid
                    print e
                    miss_gstd_num+=1
                    continue 
            print "%d query PMIDs miss MeSH terms"%miss_gstd_num
            cPickle.dump(self.query_std_mesh_dict, file(self.qpmid_std_mesh_file,"w"))

    def load_top_cands(self):
        try:
            self.top_cands_dict = cPickle.load(file(self.top_cands_file))
        except:
            print "Top candidates not ready. Run candidates_tops_from_BM25_and_queries.py"        

    def eval_top_candidates(self):
        '''Evaluate average coverage of top candidates from queries and BM25KNN neighbors'''
        joint_coverage_list = []
        joint_cand_num_list = []        
        query_pmid_list = self.query_std_mesh_dict.keys()

        for q_pmid in query_pmid_list:
            gstd = self.query_std_mesh_dict.get(q_pmid)
            tops = self.top_cands_dict.get(q_pmid)

            overlap = set(gstd)&set(tops)
            joint_cand_num_list.append(len(tops))
            try:
                coverage = len(overlap)*1.0/len(gstd)
                joint_coverage_list.append(coverage)
            except Exception as e:
                print q_pmid
                print e
        print "%s PMIDs' joint candidates average coverage on MEDLINE %s: "%(self.query_dataset, self.period), sum(joint_coverage_list)*1.0/len(joint_coverage_list)
        print "Average num of joint candidates: ", sum(joint_cand_num_list)*1.0/len(joint_cand_num_list)
        print "Valid joint candidate number: ",len(joint_coverage_list)

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

    L1000_top_cands = Eval_top_candidates_from_BM25_and_queries(data_dir, data2_dir, knn_base_dir, analysis_dir, time_span, pmid_year_dict_file, top_k)
    L1000_top_cands.run()