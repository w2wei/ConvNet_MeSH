'''
    Analyze the coverage of top candidates from KNN and query in different settings

    Created on August 5, 2016
    Updated on August 5, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from corpus_medline_retrieve import Retrieve
from corpus_medline_build import GetMEDLINE
from corpus_medline_index import extract_TiAb_from_MEDLINE, extract_MeSH_from_MEDLINE, Index
from corpus_medline_IndexFiles import IndexFiles
from collections import defaultdict
from knn_data import Data
from collections import Counter
from Bio import Entrez
from operator import itemgetter
Entrez.email = "granitedewint@gmail.com"

class NotImplementedError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

class Top_candidate_coverage(object):
    ''' 
        Find the coverage of top candidates from different combinations 
        This class is based on corpus_medline_knn_mesh_coverage.Earlier_pmids_as_ref
    '''
    def __init__(self):
        ## corpus period 1995-1997
        self.top_threshold = 100
        self.period = "1995_1997"
        self.query = '''("1995/01/01"[Date - MeSH] : "1997/12/31"[Date - MeSH])'''
        self.data_dir = '/home/w2wei/data'
        self.index_dir = os.path.join(self.data_dir, "index")
        self.index_file = "latest_3M.index"
        self.medline_1997_index_file = "medline_%s.index"%self.period
        self.out_dir = os.path.join(self.data_dir, "latest_3M_analysis")
        self.knn_file = os.path.join(self.out_dir, "NLM2007_as_query_knn_from_%s.pkl"%self.period) ## KNN results output file
        self.pmid_1997_dir = os.path.join(self.out_dir, "pmid_%s"%self.period)
        self.medline_1997_dir = os.path.join(self.out_dir, "medline_%s"%self.period)
        self.tiab_1997_dir = os.path.join(self.out_dir, "tiab_%s"%self.period)
        self.mesh_1997_dir = os.path.join(self.out_dir, "mesh_%s"%self.period)
        self.nlm2007_pmid_dir = os.path.join(self.out_dir, "nlm2007_pmids")
        self.nlm2007_medline_dir = os.path.join(self.out_dir, "nlm2007_medline")
        self.nlm2007_qpmid_top_cand_file = os.path.join(self.out_dir, "nlm2007_qpmids_top_cand_from_%s.pkl"%self.period)
        self.nlm2007_qpmid_top_d2v_tfidf_cand_file = os.path.join(self.out_dir,"nlm2007_qpmids_top_d2v_tfidf_cand_from_%s.pkl"%self.period)
        if not os.path.exists(self.nlm2007_pmid_dir):
            os.makedirs(self.nlm2007_pmid_dir)
        if not os.path.exists(self.nlm2007_medline_dir):
            os.makedirs(self.nlm2007_medline_dir)
        self.nlm2007_qpmid_mesh_from_query_file = os.path.join(self.out_dir,"NLM2007_q_pmid_mesh_from_query_%s.pkl"%self.period)
        self.mmDir = os.path.join(self.data_dir, "metamap_data") ## dir for metamap input files
        self.nlm2007_metamap_dir = os.path.join(self.mmDir, "nlm2007")        
        self.metamap_out_file = os.path.join(self.nlm2007_metamap_dir, "nlm2007_pmid_cand_score_dict.pkl")

        self.query_dict = defaultdict()
        self.query_date_dict = {}
        self.knn_date_dict = {}
        self.NLM2007_pmids = []
        self.result = defaultdict()
        self.query_mesh_dict = defaultdict() ## gold standard mesh for query pmids in NLM2007
        self.coverage_list = []
        self.top_cand_dict = defaultdict()
        self.top_cand_coverage_list = []
        self.joint_coverage_list = []
        self.nlm2007_pmid_knn_dict = {}
        self.std_mesh_dict = {}
        self.d2v_tfidf_knn_mesh_dict = {}
        self.d2v_tfidf_knn_score_dict = {}
        self.top_d2v_tfidf_cand_dict = defaultdict()
        self.str_match_from_query_dict = defaultdict()
        self.raw_metamap_from_query_dict = defaultdict()
        self.metamap_from_query_dict = defaultdict()

    def run(self):
        self.load_queries_NLM2007() ## load queries from NLM2007 {q_pmid: query text}

        ## Experiments on MEDLINE 1995-1997, BM25
        self.retrieve_alternative() # to get self.result = {query pmid: [(knn pmid, sim score)]}
        self.comp_coverage()
        self.comp_candidate_weight() # to get self.top_cand_dict
        self.comp_top_BM25_KNN_candidate_coverage()
        self.load_candidates_from_query()
        self.comp_join_top_BM25_KNN_candidate_and_query_coverage()
        sys.exit(1)
        ## Experiments on MEDLINE1995-1997, D2V+TFIDF as repr
        self.load_d2v_tfidf_knn()
        self.comp_d2v_tfidf_candidate_weight()
        self.comp_top_d2v_tfidf_candidate_coverage()
        ## Experiments joining BM25 top terms, D2V+TFIDF top terms, and terms from query
        self.comp_join_top_KNN_candidate_and_query_coverage()

    def load_queries_NLM2007(self):
        raw_data_dir = os.path.join(self.data_dir,"lu_data")
        clean_dir = os.path.join(self.data_dir, "lu_data", "clean")
        NLM2007 = Data(raw_data_dir, clean_dir)
        NLM2007.nlm2007()
        self.NLM2007_pmids = NLM2007.query_pmids
        for pmid in self.NLM2007_pmids:
            title, abstract, _ = NLM2007.query_tam[pmid]
            self.query_dict[pmid] = ". ".join(title + abstract)

    def retrieve_alternative(self):
        '''Retrieve documents from MEDLINE 1997'''
        try:
            self.result = cPickle.load(file(self.knn_file))
        except:
            ret = Retrieve(self.index_dir, self.medline_1997_index_file, self.query_dict)
            self.result = ret.result
            cPickle.dump(self.result, file(self.knn_file, "w"))

    def comp_coverage(self):
        mesh_num_list = []
        for q_pmid, knn_pmid_score_pairs in self.result.iteritems():
            ## load gold standard MeSH of query pmids
            knn_pmids = [x[0] for x in knn_pmid_score_pairs]
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load mesh of knn pmids
            knn_mesh_list = []
            ## exclude q_pmid from knn_pmids
            knn_pmids = list(set(knn_pmids)-set([q_pmid]))            
            for k_pmid in knn_pmids:
                k_mesh = file(os.path.join(self.mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                knn_mesh_list+=k_mesh
            knn_mesh = filter(None, list(set(knn_mesh_list)))
            mesh_num_list.append(len(knn_mesh))
            ## compute coverage
            overlap = set(gstd)&set(knn_mesh)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' BM25 KNN average coverage on MEDLINE %s: "%self.period, sum(self.coverage_list)*1.0/len(self.coverage_list), len(self.coverage_list)
        print "NLM2007 PMIDs BM25 KNN average number during %s"%self.period, sum(mesh_num_list)*1.0/len(mesh_num_list)

    def comp_candidate_weight(self):
        '''compute weights of candidates from BM25 KNN papers'''
        try:
            self.top_cand_dict = cPickle.load(file(self.nlm2007_qpmid_top_cand_file))
        except:
            for q_pmid, knn_pmid_score_pairs in self.result.iteritems():
                ## load gold standard MeSH of query pmids
                gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
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
                    k_mesh_list = file(os.path.join(self.mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                    k_mesh_score_list = [(k_mesh, k_score) for k_mesh in k_mesh_list]
                    knn_mesh_score_list+=k_mesh_score_list
                
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
            cPickle.dump(self.top_cand_dict, file(self.nlm2007_qpmid_top_cand_file,"w"))

    def comp_top_BM25_KNN_candidate_coverage(self):
        '''coverage of top candidates from BM25 KNN at different thresholds'''
        print "self.top_cand_dict size: ", len(self.top_cand_dict)
        ## load mesh for pmids in NLM2007
        for q_pmid in self.result.keys():
            ## load gold standard MeSH of query pmids
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load top candidates
            top_cands = self.top_cand_dict.get(q_pmid)[:self.top_threshold]
            ## compute coverage
            overlap = set(gstd)&set(top_cands)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.top_cand_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' top %d candidates average coverage on MEDLINE %s: "%(self.top_threshold, self.period), sum(self.top_cand_coverage_list)*1.0/len(self.top_cand_coverage_list), len(self.top_cand_coverage_list)

    def load_candidates_from_query(self):
        self.str_match_from_query_dict = cPickle.load(file(self.nlm2007_qpmid_mesh_from_query_file))
        self.raw_metamap_from_query_dict = cPickle.load(file(self.metamap_out_file)) ## {pmid:[(term, score)]}
        for pmid, pairList in self.str_match_from_query_dict.iteritems():
            self.metamap_from_query_dict[pmid]=[x[0] for x in pairList]

    def comp_join_top_BM25_KNN_candidate_and_query_coverage(self):
        count_str_mt_and_top_knn = []
        count_mm_and_top_knn = []
        count_str_mt_mm_top_knn = []

        joint_str_mt_and_top_knn = []
        joint_mm_and_top_knn = []
        joint_str_mt_mm_top_knn = []        
        
        pmidList = self.str_match_from_query_dict.keys()
        
        for q_pmid in pmidList:
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            top_knn_cands = self.top_cand_dict.get(q_pmid)[:self.top_threshold]
            cands_str_match = self.str_match_from_query_dict.get(q_pmid) ## candidates from string matching from queries
            cands_metamap = self.metamap_from_query_dict.get(q_pmid) ## candidates from metamap from queries
            
            # 1
            str_mt_and_top_knn = set(top_knn_cands)|set(cands_str_match)
            # 2
            mm_and_top_knn = set(cands_metamap)|set(top_knn_cands)
            # 3
            str_mt_mm_top_knn = set(cands_metamap)|set(top_knn_cands)|set(cands_str_match)

            count_str_mt_and_top_knn.append(len(str_mt_and_top_knn))
            count_mm_and_top_knn.append(len(mm_and_top_knn))
            count_str_mt_mm_top_knn.append(len(str_mt_mm_top_knn))

            overlap_str_mt_and_top_knn = set(gstd)&set(str_mt_and_top_knn)
            try:
                coverage_str_mt_and_top_knn = len(overlap_str_mt_and_top_knn)*1.0/len(gstd)
                joint_str_mt_and_top_knn.append(coverage_str_mt_and_top_knn)
            except Exception as e:
                pass

            overlap_mm_and_top_knn = set(gstd)&set(mm_and_top_knn)
            try:
                coverage_mm_and_top_knn = len(overlap_mm_and_top_knn)*1.0/len(gstd)
                joint_mm_and_top_knn.append(coverage_mm_and_top_knn)
            except Exception as e:
                pass

            overlap_str_mt_mm_top_knn = set(gstd)&set(str_mt_mm_top_knn)
            try:
                coverage_str_mt_mm_top_knn = len(overlap_str_mt_mm_top_knn)*1.0/len(gstd)
                joint_str_mt_mm_top_knn.append(coverage_str_mt_mm_top_knn)
            except Exception as e:
                pass                

        print "NLM2007 PMIDs' joint candidates top %d BM25 KNN and String match candidates average coverage on MEDLINE %s: "%(self.top_threshold, self.period), \
        sum(joint_str_mt_and_top_knn)*1.0/len(joint_str_mt_and_top_knn), len(joint_str_mt_and_top_knn)
        print "Average num: ", sum(count_str_mt_and_top_knn)*1.0/len(count_str_mt_and_top_knn)
        print
        print "NLM2007 PMIDs' joint candidates top %d BM25 KNN and MetaMap candidates average coverage on MEDLINE %s: "%(self.top_threshold, self.period), \
        sum(joint_mm_and_top_knn)*1.0/len(joint_mm_and_top_knn), len(joint_mm_and_top_knn)
        print "Average num: ", sum(count_mm_and_top_knn)*1.0/len(count_mm_and_top_knn)
        print
        print "NLM2007 PMIDs' joint candidates top %d BM25 KNN and String match and MetaMap candidates average coverage on MEDLINE %s: "%(self.top_threshold, self.period), \
        sum(joint_str_mt_mm_top_knn)*1.0/len(joint_str_mt_mm_top_knn), len(joint_str_mt_mm_top_knn)
        print "Average num: ", sum(count_str_mt_mm_top_knn)*1.0/len(count_str_mt_mm_top_knn)

    def load_d2v_tfidf_knn(self):
        '''Load KNN pmids for NLM2007 queries based on d2v+tfidf'''
        d2v_tfidf_dir = os.path.join(self.data_dir, "medline_%s_d2v_tfidf"%self.period)
        nlm2007_pmid_knn_dict_file = os.path.join(d2v_tfidf_dir, "NLM2007_knn_dict_d2v+tfidf.pkl")
        self.nlm2007_pmid_knn_dict = cPickle.load(file(nlm2007_pmid_knn_dict_file))
        d2v_tfidf_knn_coverage_list = []
        for q_pmid, knn_pmid_score_pairs in self.nlm2007_pmid_knn_dict.iteritems():
            knn_pmid_score_pairs.sort(key=itemgetter(1),reverse=True)
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            self.std_mesh_dict[q_pmid] = gstd

            knn_pmids = [x[0] for x in knn_pmid_score_pairs[:50]]
            knn_mesh_list = []
            knn_pmids = list(set(knn_pmids)-set([q_pmid]))
            for k_pmid in knn_pmids:
                k_mesh = file(os.path.join(self.mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                knn_mesh_list+=k_mesh
            knn_mesh = filter(None, list(set(knn_mesh_list)))
            self.d2v_tfidf_knn_mesh_dict[q_pmid]=knn_mesh
            ## coverage 
            overlap = set(gstd)&set(knn_mesh)
            coverage = len(overlap)*1.0/len(gstd)
            d2v_tfidf_knn_coverage_list.append(coverage)
        print "Average coverage of all d2v+tfidf KNN MeSH: ", sum(d2v_tfidf_knn_coverage_list)*1.0/len(d2v_tfidf_knn_coverage_list)

    def comp_d2v_tfidf_candidate_weight(self):
        '''compute weights of candidates from d2v+tfidf KNN papers'''
        try:
            self.top_d2v_tfidf_cand_dict = cPickle.load(file(self.nlm2007_qpmid_top_d2v_tfidf_cand_file))
        except:
            for q_pmid, knn_pmid_score_pairs in self.nlm2007_pmid_knn_dict.iteritems():
                ## load gold standard MeSH of query pmids
                gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
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
                    k_mesh_list = file(os.path.join(self.mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                    k_mesh_score_list = [(k_mesh, k_score) for k_mesh in k_mesh_list]
                    knn_mesh_score_list+=k_mesh_score_list
                
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
                self.top_d2v_tfidf_cand_dict[q_pmid]=top_candidates
            cPickle.dump(self.top_d2v_tfidf_cand_dict, file(self.nlm2007_qpmid_top_d2v_tfidf_cand_file,"w"))

    def comp_top_d2v_tfidf_candidate_coverage(self):
        '''coverage of top d2v+tfidf candidates at different thresholds'''
        print "self.top_d2v_tfidf_cand_dict size: ", len(self.top_d2v_tfidf_cand_dict)
        ## load mesh for pmids in NLM2007
        top_d2v_tfidf_cand_coverage_list = []
        for q_pmid in self.result.keys():
            ## load gold standard MeSH of query pmids
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load top candidates
            top_cands = self.top_d2v_tfidf_cand_dict.get(q_pmid)[:]
            ## compute coverage
            overlap = set(gstd)&set(top_cands)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                top_d2v_tfidf_cand_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' top d2v+tfidf candidates average coverage on MEDLINE %s: "%self.period, sum(top_d2v_tfidf_cand_coverage_list)*1.0/len(top_d2v_tfidf_cand_coverage_list), len(top_d2v_tfidf_cand_coverage_list)

    def comp_join_top_KNN_candidate_and_query_coverage(self):
        '''BM25 KNN, d2v+tfidf cosine KNN, query'''
        cand_from_query_dict = cPickle.load(file(self.nlm2007_qpmid_mesh_from_query_file))
        joint_cand_num_list = []
        top_k = 100
        joint_all_coverage_list = []
        print "top %d candidates"%top_k
        for q_pmid in cand_from_query_dict.keys():
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            top_knn_cands = self.top_cand_dict.get(q_pmid)[:top_k]
            top_d2v_tfidf_knn_cands = self.top_d2v_tfidf_cand_dict.get(q_pmid)[:top_k]
            # cands_from_query = []
            cands_from_query = cand_from_query_dict.get(q_pmid)
            joint_set = set(top_knn_cands)|set(cands_from_query)|set(top_d2v_tfidf_knn_cands)
            overlap = set(gstd)&set(joint_set)
            joint_cand_num_list.append(len(joint_set))
            try:
                coverage = len(overlap)*1.0/len(gstd)
                joint_all_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' joint all candidates average coverage on MEDLINE %s: "%self.period, sum(joint_all_coverage_list)*1.0/len(joint_all_coverage_list), len(joint_all_coverage_list)
        print "Average num of joint all candidates: ", sum(joint_cand_num_list)*1.0/len(joint_cand_num_list)

    def save_NLM2007_query_pmids(self):
        fout = file(os.path.join(self.pmid_1997_dir, "NLM2007_query_pmid.txt"),"w")
        fout.write("\n".join(self.NLM2007_pmids))

    def get_PMID(self):
        """Retrieve PMC ID"""
        if not os.path.exists(self.pmid_1997_dir):
            os.makedirs(self.pmid_1997_dir)
        retstart = 0
        step = 5000 # restricted by NCBI
        i=0
        newPMID=step
        count=0

        while newPMID>=step:
            handle=Entrez.esearch(db='pubmed',term=self.query,retstart=i*step,retmax=step)
            record=Entrez.read(handle)
            newPMID=len(record["IdList"])
            count+=newPMID
            i+=1
            fout = file(os.path.join(self.pmid_1997_dir, "%d.txt"%i),"w")
            fout.write("\n".join(record["IdList"]))
            if count%1000 == 0:
                print "Current PMID count: ", count
        print "Total PMID count: ",count        


if __name__=="__main__":
    exp1 = Top_candidate_coverage()
    exp1.run()
    
