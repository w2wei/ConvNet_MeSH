'''
    Analyze the average precision of BM25 KNN candidates from various corpora of different time

    Created on July 29, 2016
    Updated on August 26, 2016
    @author: Wei Wei

'''

import os,sys,random,cPickle, time, datetime
from candidates_retrieve import Retrieve
from candidates_build import GetMEDLINE
from candidates_index import extract_TiAb_from_MEDLINE, extract_MeSH_from_MEDLINE, Index
from candidates_IndexFiles import IndexFiles
from collections import defaultdict
from knn_data import Data
from collections import Counter
from Bio import Entrez
from operator import itemgetter
Entrez.email = "granitedewint@gmail.com"

class NotImplementedError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

class Earlier_pmids_as_ref(object):
    '''Use NLM2007 as query, use a customized corpus '''
    def __init__(self, data_dir, out_dir, time_span):
        self.period = time_span 
        # self.query = '''("1995/01/01"[Date - MeSH] : "1997/12/31"[Date - MeSH])'''
        ## input dirs and files
        self.data_dir = data_dir
        self.index_dir = os.path.join(self.data_dir, "index")
        self.index_file = "medline_%s.index"%self.period  ## self.index_file = "latest_3M.index"

        self.pmid_1997_dir = os.path.join(self.out_dir, "pmid_%s"%self.period)
        self.medline_1997_dir = os.path.join(self.out_dir, "medline_%s"%self.period)
        self.tiab_1997_dir = os.path.join(self.out_dir, "tiab_%s"%self.period)
        self.mesh_1997_dir = os.path.join(self.out_dir, "mesh_%s"%self.period)

        ## output dirs and files
        self.out_dir = out_dir# os.path.join(self.data_dir, "latest_3M_analysis")
        self.query_dataset = os.path.basename(out_dir).split("_")[0]
        print "query_dataset ", self.query_dataset
        self.knn_file = os.path.join(self.out_dir, "%_as_query_knn_from_MEDLINE_%s.pkl"%(self.query_dataset, self.period)) ## KNN results output file


        self.nlm2007_pmid_dir = os.path.join(self.out_dir, "nlm2007_pmids")
        self.nlm2007_medline_dir = os.path.join(self.out_dir, "nlm2007_medline")
        self.nlm2007_qpmid_top_cand_file = os.path.join(self.out_dir, "nlm2007_qpmids_top_cand_from_%s.pkl"%self.period)
        self.nlm2007_qpmid_top_d2v_tfidf_cand_file = os.path.join(self.out_dir,"nlm2007_qpmids_top_d2v_tfidf_cand_from_%s.pkl"%self.period)
        if not os.path.exists(self.nlm2007_pmid_dir):
            os.makedirs(self.nlm2007_pmid_dir)
        if not os.path.exists(self.nlm2007_medline_dir):
            os.makedirs(self.nlm2007_medline_dir)

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

    def run(self):
        self.load_queries_NLM2007()
        ## Experiments on NLM2007
        self.collect_medline_NLM2007()
        self.get_knn_publication_date()
        ## Experiments on alternative datasets, BM25
        self.collect_medline_alternative()
        self.index_alternative()
        self.retrieve_alternative()
        # self.comp_coverage()
        self.comp_candidate_weight()
        self.comp_top_candidate_coverage()
        self.comp_join_top_candidate_and_query_coverage()
        ## Experiments on MEDLINE1995-1997, D2V+TFIDF as repr
        self.load_d2v_tfidf_knn()
        self.comp_d2v_tfidf_candidate_weight()
        self.comp_top_d2v_tfidf_candidate_coverage()
        ## Experiments joining BM25 top terms, D2V+TFIDF top terms, and terms from query
        self.comp_join_top_KNN_candidate_and_query_coverage()

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

    def load_queries_NLM2007(self):
        raw_data_dir = os.path.join(self.data_dir,"lu_data")
        clean_dir = os.path.join(self.data_dir, "lu_data", "clean")
        NLM2007 = Data(raw_data_dir, clean_dir)
        NLM2007.nlm2007()
        self.NLM2007_pmids = NLM2007.query_pmids
        NLM2007_query_pmid_knn_pmid_dict = NLM2007.nbr_dict
        for pmid in self.NLM2007_pmids:
            title, abstract, _ = NLM2007.query_tam[pmid]
            self.query_dict[pmid] = ". ".join(title + abstract)

        fout = file(os.path.join(self.nlm2007_pmid_dir,"nlm2007_query_pmids.txt"),"w")
        fout.write("\n".join(self.NLM2007_pmids))
        NLM2007_knn_pmid_list = []
        for q_pmid, knn_pmid_tpl_list in NLM2007_query_pmid_knn_pmid_dict.iteritems():
            NLM2007_knn_pmid_list += [item[0] for item in knn_pmid_tpl_list]
        fout2 = file(os.path.join(self.nlm2007_pmid_dir, "nlm2007_prc_knn_pmids.txt"),"w")
        fout2.write("\n".join(NLM2007_knn_pmid_list))

    def collect_medline_NLM2007(self):
        '''collect MEDLINE of query PMIDs'''
        if not os.listdir(self.nlm2007_medline_dir):
            getmed = GetMEDLINE(self.nlm2007_pmid_dir, self.nlm2007_medline_dir)
            getmed.get_medline()

    def get_knn_publication_date(self):
        medline_file_list = ["nlm2007_prc_knn_pmids.txt", "nlm2007_query_pmids.txt"]
        date_dict_list = [self.knn_date_dict, self.query_date_dict]

        for doc_idx in range(len(medline_file_list)):
            fin = filter(None, file(os.path.join(self.nlm2007_medline_dir,medline_file_list[doc_idx])).read().split("\n\n"))
            print "rec num: ", len(fin)
            for rec in fin:
                rec = rec.split("\n")
                pmid=''
                date=''
                try:
                    for line in rec:
                        if line.startswith('PMID-'):
                            pmid = line.split('PMID-')[1].strip()
                        if line.startswith('MHDA-'):
                            date = line.split('MHDA-')[1].strip()[:4]
                    date_dict_list[doc_idx][pmid]=date
                except Exception as e:
                    print e
                    pass
        knn_dates = self.knn_date_dict.values()
        knn_date_dist = Counter(knn_dates)
        # print knn_date_dist

    def collect_medline_alternative(self):
        '''Select alternative MEDLINE corpus for NLM2007 query PMIDs'''
        print "Query: ", self.query
        if not (os.path.exists(self.pmid_1997_dir) and os.listdir(self.pmid_1997_dir)):
            self.get_PMID()
            self.save_NLM2007_query_pmids()
        if not (os.path.exists(self.medline_1997_dir) and os.listdir(self.medline_1997_dir)):
            getmed = GetMEDLINE(self.pmid_1997_dir, self.medline_1997_dir)
            getmed.get_medline()
            print "MEDLINE raw ready"
            print "Input dir: ", self.pmid_1997_dir, " doc num: ", len(os.listdir(self.pmid_1997_dir))
            print "Output dir: ", self.medline_1997_dir, " doc num: ", len(os.listdir(self.medline_1997_dir))

    def index_alternative(self):
        '''Extract titles and abstracts from 1997 MEDLINE and index them'''
        ## extract titles and abstracts from raw MEDLINE
        if not os.path.exists(self.tiab_1997_dir):
            os.makedirs(self.tiab_1997_dir)
        if not os.listdir(self.tiab_1997_dir):
            extract_TiAb_from_MEDLINE(self.medline_1997_dir, self.tiab_1997_dir)
        print "TiAb ready"
        if not os.path.exists(self.mesh_1997_dir):
            os.makedirs(self.mesh_1997_dir)
        if not os.listdir(self.mesh_1997_dir):
            extract_MeSH_from_MEDLINE(self.medline_1997_dir, self.mesh_1997_dir)
        print "MeSH ready"
        if not os.path.exists(os.path.join(self.index_dir, self.index_file)):
            raise NotImplementedError("Run corpus_medline_index.py to index MEDLINE %s corpus."%self.period)

    def retrieve_alternative(self):
        '''Retrieve documents from MEDLINE 1997'''
        try:
            self.result = cPickle.load(file(self.knn_file))
        except:
            ret = Retrieve(self.index_dir, self.index_file, self.query_dict)
            self.result = ret.result
            cPickle.dump(self.result, file(self.knn_file, "w"))

    def comp_coverage(self):
        ## load mesh for pmids in NLM2007
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
        print "NLM2007 PMIDs' average coverage on MEDLINE %s: "%self.period, sum(self.coverage_list)*1.0/len(self.coverage_list), len(self.coverage_list)
        print "NLM2007 PMIDs average number of KNN MeSH during %s"%self.period, sum(mesh_num_list)*1.0/len(mesh_num_list)

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

    def comp_top_candidate_coverage(self):
        '''coverage of top candidates at different thresholds'''
        print "self.top_cand_dict size: ", len(self.top_cand_dict)
        ## load mesh for pmids in NLM2007
        for q_pmid in self.result.keys():
            ## load gold standard MeSH of query pmids
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            ## load top candidates
            top_cands = self.top_cand_dict.get(q_pmid)
            ## compute coverage
            overlap = set(gstd)&set(top_cands)
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.top_cand_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' top candidates average coverage on MEDLINE %s: "%self.period, sum(self.top_cand_coverage_list)*1.0/len(self.top_cand_coverage_list), len(self.top_cand_coverage_list)

    def comp_join_top_candidate_and_query_coverage(self):
        nlm2007_qpmid_mesh_from_query_file = os.path.join(self.out_dir,"NLM2007_q_pmid_mesh_from_query_%s.pkl"%self.period)
        cand_from_query_dict = cPickle.load(file(nlm2007_qpmid_mesh_from_query_file))
        joint_cand_num_list = []
        top_k = 100
        print "top %d candidates"%top_k
        for q_pmid in cand_from_query_dict.keys():
            gstd = filter(None, file(os.path.join(self.mesh_1997_dir,"%s.txt"%q_pmid)).read().split('\n'))
            top_knn_cands = self.top_cand_dict.get(q_pmid)[:top_k]
            cands_from_query = cand_from_query_dict.get(q_pmid)
            joint_set = set(top_knn_cands)|set(cands_from_query)
            overlap = set(gstd)&set(joint_set)
            joint_cand_num_list.append(len(joint_set))
            try:
                coverage = len(overlap)*1.0/len(gstd)
                self.joint_coverage_list.append(coverage)
            except Exception as e:
                pass
        print "NLM2007 PMIDs' joint candidates average coverage on MEDLINE %s: "%self.period, sum(self.joint_coverage_list)*1.0/len(self.joint_coverage_list), len(self.joint_coverage_list)
        print "Average num of joint candidates: ", sum(joint_cand_num_list)*1.0/len(joint_cand_num_list)

    def comp_join_top_KNN_candidate_and_query_coverage(self):
        '''BM25 KNN, d2v+tfidf cosine KNN, query'''
        nlm2007_qpmid_mesh_from_query_file = os.path.join(self.out_dir,"NLM2007_q_pmid_mesh_from_query_%s.pkl"%self.period)
        cand_from_query_dict = cPickle.load(file(nlm2007_qpmid_mesh_from_query_file))
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
    time_span = sys.argv[1] # "1995_1997"
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    time_span = "%s_%s"%(startyear, endyear)
    
    data_dir = '/home/w2wei/data'
    out_dir = os.path.join(data_dir, 'analysis', 'L1000_query_PMID_on_MEDLINE_%s'%time_span)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    L1000_analysis = Earlier_pmids_as_ref(data_dir, out_dir, time_span)
    L1000_analysis.run()
    