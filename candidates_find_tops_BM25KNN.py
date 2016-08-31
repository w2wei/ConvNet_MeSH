'''
    Find top candiates from BM25KNN neighbors

    Created on August 30, 2016
    Updated on August 30, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from collections import defaultdict
from knn_data import Data
from collections import Counter
from operator import itemgetter

class BM25KNN_top_candidates(object):
    '''Use NLM2007 as query, use a customized corpus '''
    def __init__(self, data_dir, data2_dir, knn_base_dir, out_dir, time_span, pmid_year_dict_file):
        self.period = time_span 
        self.query_dataset = os.path.basename(out_dir).split("_")[0] # L1000

        ## input dirs and files
        self.data_dir = data_dir ## disk 1
        self.data2_dir = data2_dir ## disk 2
        self.index_dir = os.path.join(self.data2_dir, "index")
        self.index_file = "medline_%s.index"%self.period  ## self.index_file = "latest_3M.index"
        self.pmid_year_dict = cPickle.load(file(pmid_year_dict_file))
        
        self.pmid_base_dir = os.path.join(self.data2_dir, "pmid_docs_by_year")
        self.medline_base_dir = os.path.join(self.data2_dir, "medline_docs_by_year")
        self.tiab_base_dir = os.path.join(self.data2_dir, "tiab_by_year")
        self.mesh_base_dir = os.path.join(self.data2_dir, "mesh_by_year")
        self.knn_dir = os.path.join(knn_base_dir, "%s_on_MEDLINE_%s"%(self.query_dataset, self.period))
        startyear, endyear = time_span.split("_")
        self.years = range(int(startyear), int(endyear)+1)
        self.query_medline_dirs = [os.path.join(self.medline_base_dir, str(year)) for year in self.years]
        self.knn_file = os.path.join(self.knn_dir, "%s_knn_on_MEDLINE_%s.pkl"%(self.query_dataset, self.period)) ## KNN results output file

        ## output dirs and files
        self.out_dir = out_dir# os.path.join(self.data2_dir, "latest_3M_analysis")
        self.query_pmid_dir = os.path.join(self.out_dir, "%s.pmids"%self.query_dataset)
        self.qpmid_top_cand_file = os.path.join(self.out_dir, "%s_qpmid_top_cand_on_MEDLINE_%s.pkl"%(self.query_dataset, self.period))

        ## class variables
        self.query_dict = defaultdict()
        self.query_pmids = []
        self.bm25knn_result = defaultdict()
        self.top_cand_dict = defaultdict()

    def run(self):
        self.load_lu_queries()
        self.verify_query_medline()
        self.load_BM25KNN_docs()
        self.get_top_BM25KNN_candidates()

    def load_lu_queries(self):
        '''Load queries from Lu's data'''
        raw_data_dir = os.path.join(self.data_dir,"lu_data")
        clean_dir = os.path.join(self.data_dir, "lu_data", "clean")

        if self.query_dataset=="L1000":
            query_data = Data(raw_data_dir, clean_dir)
            query_data.large1000()

        if self.query_dataset=="NLM2007":
            query_data = Data(raw_data_dir, clean_dir)
            query_data.nlm2007()

        self.query_pmids = query_data.query_pmids
        query_pmid_knn_pmid_dict = query_data.nbr_dict
        for pmid in self.query_pmids:
            title, abstract, _ = query_data.query_tam[pmid]
            self.query_dict[pmid] = ". ".join(title + abstract)

        if not os.path.exists(self.query_pmid_dir):
            os.makedirs(self.query_pmid_dir)
        fout = file(os.path.join(self.query_pmid_dir,"%s_query_pmids.txt"%self.query_dataset),"w")
        fout.write("\n".join(self.query_pmids))

        knn_pmid_list = []
        for q_pmid, knn_pmid_tpl_list in query_pmid_knn_pmid_dict.iteritems():
            knn_pmid_list += [item[0] for item in knn_pmid_tpl_list]
        fout2 = file(os.path.join(self.query_pmid_dir, "%s_prc_knn_pmids.txt"%self.query_dataset),"w")
        fout2.write("\n".join(knn_pmid_list))

    def verify_query_medline(self):
        '''verify the existence of MEDLINE of query PMIDs'''
        for medline_dir in self.query_medline_dirs:
            if not os.path.exists(medline_dir):
                year = os.path.basename(medline_dir)
                print "%_MEDLINE docs missing"%year
                sys.exit(1)

    def load_BM25KNN_docs(self):
        '''Load BM25KNN docs from indexed corpus'''
        try:
            self.bm25knn_result = cPickle.load(file(self.knn_file))
        except:
            print "%s missing"%self.knn_file
            sys.exit(1)

    def get_top_BM25KNN_candidates(self):
        '''compute weights of candidates from BM25 KNN papers, and get top candidates'''
        try:
            self.top_cand_dict = cPickle.load(file(self.qpmid_top_cand_file))
        except:
            miss_kmesh_num = 0
            for q_pmid, knn_pmid_score_pairs in self.bm25knn_result.iteritems():
                ## load mesh of knn pmids
                knn_mesh_score_list = []
                ## load {k_pmid: BM25 score}
                k_pmid_score_dict = dict(knn_pmid_score_pairs)
                sum_score = sum(k_pmid_score_dict.values())
                ## exclude q_pmid from knn_pmids
                knn_pmids = k_pmid_score_dict.keys()
                knn_pmids = list(set(knn_pmids)-set([q_pmid]))

                for k_pmid in knn_pmids:
                    k_score = k_pmid_score_dict.get(k_pmid)
                    try:
                        k_pmid_year = self.pmid_year_dict.get(k_pmid)
                        k_mesh_file = os.path.join(self.mesh_base_dir,k_pmid_year,"%s.txt"%k_pmid)
                        k_mesh_list = filter(None, file(k_mesh_file).read().split('\n'))
                        k_mesh_score_list = [(k_mesh, k_score) for k_mesh in k_mesh_list]
                        knn_mesh_score_list+=k_mesh_score_list
                    except Exception as e:
                        print "Missing MeSH file ", k_pmid
                        print e
                        miss_kmesh_num+=1
                
                k_mesh_score_dict = defaultdict()
                for mesh, score in knn_mesh_score_list:
                    if not k_mesh_score_dict.get(mesh):
                        k_mesh_score_dict[mesh]=[score]
                    else:
                        k_mesh_score_dict[mesh]+=[score]
                for mesh, scoreList in k_mesh_score_dict.iteritems():
                    k_mesh_score_dict[mesh]=sum(scoreList)*1.0/sum_score

                k_mesh_vocab_score_list = k_mesh_score_dict.items()
                k_mesh_vocab_score_list.sort(key=itemgetter(1),reverse=True)
                top_candidates = [x[0] for x in k_mesh_vocab_score_list]
                self.top_cand_dict[q_pmid]=top_candidates
            cPickle.dump(self.top_cand_dict, file(self.qpmid_top_cand_file,"w"))

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

    L1000_top_cands = BM25KNN_top_candidates(data_dir, data2_dir, knn_base_dir, analysis_dir, time_span, pmid_year_dict_file)
    L1000_top_cands.run()
    