'''
    Evaluate average precisions of MeSH candidates generated from BM25 KNN, D2V KNN, TFIDF KNN, D2V+TFIDF KNN, and BM25+D2V+TFIDF KNN

    Created on July 27, 2016
    Updated on August 9, 2016
    @author: Wei Wei
'''

import os, cPickle, time, re
from collections import defaultdict, Counter
from knn_data import Data
import numpy as np

class KNN_MeSH(object):
    '''Using NLM2007'''
    def __init__(self, data_name, data_dir, knn_dir, knn_3M_file, mesh_3M_dir, analysis_dir, bm25_q_pmid_knn_mesh_dict_file ,bm25_pmid_mesh_dict_file, new_repr_knn_file, new_repr_pmid_mesh_dict_file):
        self.dataDir = data_dir
        self.knnDir = knn_dir
        self.knn_data = cPickle.load(file(knn_3M_file))
        self.dataname = data_name
        self.meshDir = mesh_3M_dir
        self.analyDir = analysis_dir
        self.bm25_pmid_mesh_dict_file = bm25_pmid_mesh_dict_file
        self.new_repr_pmid_mesh_dict_file = new_repr_pmid_mesh_dict_file
        self.bm25_q_pmid_knn_mesh_dict_file = bm25_q_pmid_knn_mesh_dict_file

        ## std mesh for all pmids
        self.std_pmid_mesh_dict = defaultdict() ## save {pmid:[mesh]} for all PMIDs, including query and knn pmids

        ## for knn retrieved using new representations
        self.new_repr_knn_dict = cPickle.load(file(new_repr_knn_file))
        self.new_repr_pmid_mesh_dict = defaultdict() ## only include the query PMIDs from Lu's data, i.e. NLM2007, SMALL200, L1000, not including KNN pmids from Lu

        ## for knn retrieved using bm25
        self.bm25_qpmid_knn_pmid_dict = defaultdict() ## {query pmid: [knn pmid list]}
        self.bm25_pmid_mesh_dict = defaultdict() ##{query pmid: [knn mesh list]}

        ## gold standard
        self.N2007_query_PRC_knn_pmid_dict = defaultdict() ## Lu data {query pmid: [knn pmid]} in NLM2007
        self.N2007_all_pmid_std_mesh_dict = defaultdict() ## NLM2007 {query pmid:[std mesh]}

    def run(self):
        self.load_lu_mesh() ## mesh from gold standard
        self.load_emb_repr_knn_mesh() ## mesh from d2v/tfidf knn
        self.comp_new_repr_with_lu_data()
        self.load_bm25_knn_mesh() ## mesh from bm25 knn
        self.comp_bm25_with_lu_data()
        self.comp_joint_coverage()

    def load_lu_mesh(self):
        '''load knn mesh terms for NLM2007'''
        try:
            self.N2007_query_PRC_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"NLM2007_q_pmid_PRC_knn_pmid_dict.pkl"))) ## q_pmid:[knn pmids]
            self.N2007_all_pmid_std_mesh_dict = cPickle.load(file(os.path.join(self.analyDir,"NLM2007_q_pmid_std_mesh_dict.pkl"))) ## gold standard
        except:
            raw_data_dir = os.path.join(self.dataDir,"lu_data")
            clean_dir = os.path.join(self.dataDir, "lu_data", "clean")

            NLM2007 = Data(raw_data_dir, clean_dir)
            NLM2007.nlm2007()

            N2007_pmids = NLM2007.query_pmids

            for pmid in N2007_pmids:
                if self.N2007_all_pmid_std_mesh_dict.get(pmid)==None:
                    try:
                        query_pmid_mesh = self.mesh_parser4Lu_data(NLM2007.query_tam[pmid][2])
                        self.N2007_all_pmid_std_mesh_dict[pmid]=query_pmid_mesh
                    except:
                        self.N2007_all_pmid_std_mesh_dict[pmid]=[]

                knn_pmids = [x[0] for x in NLM2007.nbr_dict[pmid]]
                self.N2007_query_PRC_knn_pmid_dict[pmid] = knn_pmids

                for k_pmid in knn_pmids:
                    if self.N2007_all_pmid_std_mesh_dict.get(k_pmid)==None:
                        try:
                            knn_pmid_mesh = self.mesh_parser4Lu_data(NLM2007.nbr_tam[k_pmid][2])
                            self.N2007_all_pmid_std_mesh_dict[k_pmid] = knn_pmid_mesh
                        except:
                            self.N2007_all_pmid_std_mesh_dict[k_pmid] = []
            cPickle.dump(self.N2007_query_PRC_knn_pmid_dict, file(os.path.join(self.analyDir,"NLM2007_q_pmid_PRC_knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.N2007_all_pmid_std_mesh_dict, file(os.path.join(self.analyDir,"NLM2007_q_pmid_std_mesh_dict.pkl"),"w"))

    def load_emb_repr_knn_mesh(self):
        '''Load MeSH terms from KNN papers based on d2v and/or tfidf'''
        try:
            self.new_repr_pmid_mesh_dict = cPickle.load(file(self.new_repr_pmid_mesh_dict_file))
        except:
            for q_pmid, knn_list in self.new_repr_knn_dict.iteritems():
                knn_pmids = [tpl[0] for tpl in knn_list][:50] ## top 50 knn pmids
                knn_mesh_list = []
                for k_pmid in knn_pmids:
                    raw_k_mesh_list = filter(None, file(os.path.join(self.meshDir, "%s.txt"%k_pmid)).read().split("\n"))
                    if raw_k_mesh_list:
                        k_mesh_list = [re.sub("[-*&]"," ",mesh).strip("*").lower() for mesh in raw_k_mesh_list]
                        knn_mesh_list+=k_mesh_list
                self.new_repr_pmid_mesh_dict[q_pmid]=list(set(knn_mesh_list))
            cPickle.dump(self.new_repr_pmid_mesh_dict, file(self.new_repr_pmid_mesh_dict_file,"wb"), protocol=cPickle.HIGHEST_PROTOCOL)
            print "self.new_repr_pmid_mesh_dict updated. ", len(self.new_repr_pmid_mesh_dict)

    def _load_mesh(self, mesh_file):
        terms = file(mesh_file).read().split("\n")
        return terms

    def load_bm25_knn_mesh(self):
        self.std_pmid_mesh_dict = cPickle.load(file(os.path.join(self.analyDir,"pmid_mesh_dict.pkl")))# if this is not working, run corpus_medline_knn_mesh.py
        try:
            self.bm25_qpmid_knn_pmid_dict = cPickle.load(file(self.bm25_pmid_mesh_dict_file))
            self.bm25_pmid_mesh_dict = cPickle.load(file(self.bm25_q_pmid_knn_mesh_dict_file))
        except Exception as e:
            print e
            self.bm25_qpmid_knn_pmid_dict = self.knn_data[self.dataname] # {query pmid:[knn pmids]} by bm25
            self.bm25_pmid_mesh_dict

            for q_pmid, knn_pmids in self.bm25_qpmid_knn_pmid_dict.iteritems():
                local_knn_mesh = []
                for k_pmid in knn_pmids:
                    local_knn_mesh += self.std_pmid_mesh_dict.get(k_pmid)
                self.bm25_pmid_mesh_dict[q_pmid]=list(set(local_knn_mesh))
            cPickle.dump(self.bm25_qpmid_knn_pmid_dict, file(self.bm25_pmid_mesh_dict_file,"w"))
            cPickle.dump(self.bm25_pmid_mesh_dict, file(self.bm25_q_pmid_knn_mesh_dict_file, "w"))
            print "bm25_qpmid_knn_pmid_dict and bm25_pmid_mesh_dict updated."

    def mesh_parser4Lu_data(self, text):
        text = re.sub("[-*&]"," ",text)
        mhList = text.split("!")
        mhList = [mh.strip("*") for mh in mhList]
        return mhList

    def comp_new_repr_with_lu_data(self):
        self.comp_new_repr_mesh_coverage_per_q_pmid()

    def comp_new_repr_mesh_coverage_per_q_pmid(self):
        '''given a pmid, compare its mesh from lu data and latest 3M using the new repr d2v/tfidf/d2v+tfidf'''
        q_pmid_list = self.new_repr_knn_dict.keys() 
        print "q_pmid ", len(q_pmid_list)
        coverage_list = []
        for q_pmid in q_pmid_list:
            gstd = self.lu_pmid_mesh_dict.get(q_pmid)## gold standard
            new_repr_knn_mesh = self.new_repr_pmid_mesh_dict.get(q_pmid)
            try:
                overlap = set(gstd)&set(new_repr_knn_mesh)
            except:
                overlap = set([])
            coverage = len(overlap)*1.0/len(gstd)
            coverage_list.append(coverage)
        print "Average coverage from d2v ", sum(coverage_list)*1.0/len(coverage_list), len(coverage_list)

  N2007_qdef coPRC_mp_bm25_with_lu_data(self):
        self.comp_bm25_mesh_coverage_per_query_pmid()
        
    def comp_bm25_mesh_coverage_per_query_pmid(self):
        '''given a pmid, compare its mesh from lu data and latest 3M'''
        coverage_list = []
        for pmid, knn_mesh in self.bm25_pmid_mesh_dict.iteritems():
            gstd = self.lu_pmid_mesh_dict.get(pmid)
            bm25_knn_mesh = knn_mesh
            try:
                overlap = set(gstd)&set(bm25_knn_mesh)
            except:
                overlap = set([])            
            coverage = len(overlap)*1.0/len(gstd)
            coverage_list.append(coverage)
        print "BM25 average coverage ", np.mean(coverage_list)     

  N2007_qdef coPRC_mp_joint_coverage(self):
        ## for every q_pmid, count the overlapped MeSH terms in gold standard and D2V/TFIDF and bm25
        q_pmid_list = self.new_repr_knn_dict.keys() 
        coverage_list = []
        new_repr_candidate_num_list = []
        bm25_candidate_num_list = []
        joint_candidate_num_list = []
        for q_pmid in q_pmid_list:
            gstd = self.lu_pmid_mesh_dict.get(q_pmid)## gold standard
            new_repr_knn_mesh = self.new_repr_pmid_mesh_dict.get(q_pmid) ## knn mesh from D2V/TFIDF
            bm25_knn_mesh = self.bm25_pmid_mesh_dict.get(q_pmid)
            if new_repr_knn_mesh==None:
                new_repr_knn_mesh=[]
            joint_knn_mesh = set(new_repr_knn_mesh)|set(bm25_knn_mesh)
            try:
                overlap = set(gstd)&set(joint_knn_mesh)
            except:
                overlap = set([])             
            coverage = len(overlap)*1.0/len(gstd)
            coverage_list.append(coverage)
            new_repr_candidate_num_list.append(len(new_repr_knn_mesh))
            bm25_candidate_num_list.append(len(bm25_knn_mesh))
            joint_candidate_num_list.append(len(joint_knn_mesh))
        print "Joint average coverage: ", sum(coverage_list)*1.0/len(coverage_list)
        print "q_pmid num: ", len(q_pmid_list)

ifN2007_q_name_PRC__ == "__main__":
    dataset_name = "NLM2007"
    print "Dataset ", dataset_name

    data_dir = '/home/w2wei/data'

    ## KNN dir and files
    knn_base_dir = os.path.join(data_dir, "knn")
    knn_dir = os.path.join(knn_base_dir, "knn_latest_1M")
    lu_data_BM25KNN_latest_1M_file = os.path.join(knn_dir, "lu_data_BM25_KNN.pkl") ## BM25 KNN

    ## mesh dir and files
    mesh_base_dir = os.path.join(data_dir, "mesh")
    mesh_dir = os.path.join(mesh_base_dir, "mesh_latest_1M")

    ## analysis dir and files
    analysis_base_dir = os.path.join(data_dir, "analysis")    
    analysis_dir = os.path.join(analysis_base_dir, "NLM2007_all_KNN_on_MEDLINE_latest_1M")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    NLM2007_q_pmid_knn_mesh_bm25 = os.path.join(analysis_dir,"N2007_query_knn_pmid_dict.pkl") ## {query pmid:[knn mesh]} for NLM2007
    NLM2007_knn_dict_bm25_file = os.path.join(analysis_dir, "N2007_query_pmid_knn_mesh_dict.pkl") ##{query pmid:[knn mesh list]} for NLM2007
    NLM2007_knn_dict_d2v_file = os.path.join(analysis_dir, "NLM2007_knn_dict_d2v.pkl") ## run candidates_d2v_tfidf_knn.py, candidates_analysis_BM25KNN_from_latest_3M.py
    NLM2007_q_pmid_mesh_dict_d2v = os.path.join(analysis_dir, "NLM2007_q_pmid_knn_mesh_d2v.pkl") 
    NLM2007_knn_dict_tfidf_file = os.path.join(analysis_dir, "NLM2007_knn_dict_tfidf.pkl") ## run candidates_d2v_tfidf_knn.py, candidates_analysis_BM25KNN_from_latest_3M.py
    NLM2007_q_pmid_mesh_dict_tfidf = os.path.join(analysis_dir, "NLM2007_q_pmid_knn_mesh_tfidf.pkl")
    NLM2007_knn_dict_d2v_tfidf_file = os.path.join(analysis_dir, "NLM2007_knn_dict_d2v+tfidf.pkl") ## run candidates_d2v_tfidf_knn.py, candidates_analysis_BM25KNN_from_latest_3M.py
    NLM2007_q_pmid_mesh_dict_d2v_tfidf = os.path.join(analysis_dir, "NLM2007_q_pmid_knn_mesh_d2v+tfidf.pkl")

    ## D2v
    print "D2V repr"
    knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, lu_data_BM25KNN_latest_1M_file, mesh_dir, analysis_dir, \
        NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
        NLM2007_knn_dict_d2v_file, NLM2007_q_pmid_mesh_dict_d2v)
    knn_mesh.run()
    print
    ## tf-idf
    print "TF-IDF repr"
    knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, lu_data_BM25KNN_latest_1M_file, mesh_dir, analysis_dir, \
        NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
        NLM2007_knn_dict_tfidf_file, NLM2007_q_pmid_mesh_dict_tfidf)
    knn_mesh.run()
    print
    ## d2v+tfidf
    print "D2V+TFIDF repr"
    knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, lu_data_BM25KNN_latest_1M_file, mesh_dir, analysis_dir, \
        NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
        NLM2007_knn_dict_d2v_tfidf_file, NLM2007_q_pmid_mesh_dict_d2v_tfidf)       
    knn_mesh.run()

    # data_dir = '/home/w2wei/data'
    # knn_dir = os.path.join(data_dir, "knn_from_clean")
    # knn_3M_file = os.path.join(knn_dir, "lu_data_knn_from_3M.pkl") ## BM25 KNN
    # latest_3M_mesh_dir = os.path.join(data_dir, 'latest_3M_mesh') ## MeSH of latest 3M PMID
    # analysis_dir = os.path.join(knn_dir, "analysis_lu_data_3M")
    # if not os.path.exists(analysis_dir):
    #     os.makedirs(analysis_dir)
    # NLM2007_q_pmid_knn_mesh_bm25 = os.path.join(analysis_dir,"N2007_query_knn_pmid_dict.pkl") ## {query pmid:[knn mesh]} for NLM2007
    # NLM2007_knn_dict_bm25_file = os.path.join(analysis_dir, "N2007_query_pmid_knn_mesh_dict.pkl") ##{query pmid:[knn mesh list]} for NLM2007

    # d2v_tfidf_dir = os.path.join(data_dir, "latest_3M_d2v_tfidf")

    # NLM2007_knn_dict_d2v_file = os.path.join(d2v_tfidf_dir, "NLM2007_knn_dict_d2v.pkl")
    # NLM2007_q_pmid_mesh_dict_d2v = os.path.join(d2v_tfidf_dir, "NLM2007_q_pmid_knn_mesh_d2v.pkl")

    # NLM2007_knn_dict_tfidf_file = os.path.join(d2v_tfidf_dir, "NLM2007_knn_dict_tfidf.pkl")
    # NLM2007_q_pmid_mesh_dict_tfidf = os.path.join(d2v_tfidf_dir, "NLM2007_q_pmid_knn_mesh_tfidf.pkl")
    
    # NLM2007_knn_dict_d2v_tfidf_file = os.path.join(d2v_tfidf_dir, "NLM2007_knn_dict_d2v+tfidf.pkl")
    # NLM2007_q_pmid_mesh_dict_d2v_tfidf = os.path.join(d2v_tfidf_dir, "NLM2007_q_pmid_knn_mesh_d2v+tfidf.pkl")

    # dataset_name = "NLM2007"
    # print "Dataset ", dataset_name
    # ## D2v
    # print "D2V repr"
    # knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, knn_3M_file, latest_3M_mesh_dir, analysis_dir, \
    #     NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
    #     NLM2007_knn_dict_d2v_file, NLM2007_q_pmid_mesh_dict_d2v)
    # knn_mesh.run()
    # print
    # ## tf-idf
    # print "TF-IDF repr"
    # knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, knn_3M_file, latest_3M_mesh_dir, analysis_dir, \
    #     NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
    #     NLM2007_knn_dict_tfidf_file, NLM2007_q_pmid_mesh_dict_tfidf)
    # knn_mesh.run()
    # print
    # ## d2v+tfidf
    # print "D2V+TFIDF repr"
    # knn_mesh = KNN_MeSH(dataset_name, data_dir, knn_dir, knn_3M_file, latest_3M_mesh_dir, analysis_dir, \
    #     NLM2007_knn_dict_bm25_file, NLM2007_q_pmid_knn_mesh_bm25, \
    #     NLM2007_knn_dict_d2v_tfidf_file, NLM2007_q_pmid_mesh_dict_d2v_tfidf)       
    # knn_mesh.run()

