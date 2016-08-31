'''
    Analyze the average precision of BM25 KNN candidates

    Created on July 29, 2016
    Updated on August 30, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from collections import defaultdict
from knn_data import Data
from collections import Counter
from operator import itemgetter
from candidates_find_tops_BM25KNN import BM25KNN_top_candidates

class Eval_BM25_KNN_candidates(BM25KNN_top_candidates):
    '''Use NLM2007 as query, use a customized corpus '''
    def __init__(self, data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file):
        super(Eval_BM25_KNN_candidates, self).__init__(data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file)
        self.qpmid_std_mesh_file = os.path.join(self.out_dir, "%s_qpmid_std_mesh.pkl"%(self.query_dataset)) ## gold standard MeSH terms of query PMIDs
        self.query_std_mesh_dict = defaultdict()

    def run(self):
        self.load_BM25KNN_docs()
        self.load_std_mesh()
        self.comp_coverage()

    def load_BM25KNN_docs(self):
        '''Load BM25KNN docs from indexed corpus'''
        try:
            self.bm25knn_result = cPickle.load(file(self.knn_file))
        except:
            print "%s missing"%self.knn_file
            sys.exit(1)

    def load_top_BM25KNN_candidates(self):
        '''loadtop candidates'''
        try:
            self.top_cand_dict = cPickle.load(file(self.qpmid_top_cand_file))
        except:
            print "Top candidates from BM25KNN are not yet ready. Run candidates_find_tops_BM25KNN.py"
            sys.exit(1)

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

    def comp_coverage(self):
        '''Compute the average precision/coverage candidates from BM25KNN docs'''
        ## load mesh for pmids in NLM2007
        knn_mesh_num_list = []
        miss_gstd_num = 0
        miss_kmesh_num = 0
        bm25knn_cand_coverage_list = []
        for q_pmid, knn_pmid_score_pairs in self.bm25knn_result.iteritems():
            ## load knn pmids
            knn_pmids = [x[0] for x in knn_pmid_score_pairs]
            ## load gold standard MeSH of query pmids
            gstd = self.query_std_mesh_dict.get(q_pmid)
            if not gstd:
                continue
            ## load mesh of knn pmids
            knn_mesh_list = []
            ## exclude q_pmid from knn_pmids
            knn_pmids = list(set(knn_pmids)-set([q_pmid]))
            for k_pmid in knn_pmids:
                try:
                    k_pmid_year = self.pmid_year_dict.get(k_pmid)
                    k_mesh_file = os.path.join(self.mesh_base_dir,k_pmid_year,"%s.txt"%k_pmid)
                    k_mesh = filter(None, file(k_mesh_file).read().split('\n'))
                    knn_mesh_list+=k_mesh
                except Exception as e:
                    print "Missing MeSH file ", k_pmid
                    print e
                    miss_kmesh_num+=1
            knn_mesh = filter(None, list(set(knn_mesh_list)))
            knn_mesh_num_list.append(len(knn_mesh))
            ## compute coverage
            overlap = set(gstd)&set(knn_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                bm25knn_cand_coverage_list.append(coverage)
            except Exception as e:
                print "BM25KNN coverage error; query PMID ", q_pmid
                print e
        print "%s PMIDs' average coverage on MEDLINE %s: "%(self.query_dataset, self.period), sum(bm25knn_cand_coverage_list)*1.0/len(bm25knn_cand_coverage_list)
        print "%s PMIDs' average number of BM25KNN MeSH on MEDLINE %s"%(self.query_dataset, self.period), sum(knn_mesh_num_list)*1.0/len(knn_mesh_num_list)
        print "Valid coverage value number: ", len(bm25knn_cand_coverage_list)

if __name__=="__main__":
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)
    
    data_dir = '/home/w2wei/data'
    data2_dir = '/home/w2wei/data2'
    analysis_dir = os.path.join(data2_dir, 'analysis', 'L1000_query_PMID_on_MEDLINE_%s'%time_span)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    util_dir = os.path.join(data2_dir, "utils")
    pmid_year_dict_file = os.path.join(util_dir, "pmid_year_dict.pkl")

    knn_base_dir = os.path.join(data2_dir, "knn")

    L1000_top_cands = Eval_BM25_KNN_candidates(data_dir, data2_dir, knn_base_dir, analysis_dir, time_span, pmid_year_dict_file)
    L1000_top_cands.run()