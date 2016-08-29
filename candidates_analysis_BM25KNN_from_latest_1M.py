'''
    Evaluate the average precision of MeSH candidates generated using BM25 KNN on the latest 3M MEDLINE. Experiments were done on L1000, NLM2007, SMALL200.
    BM25 KNN PMIDs generated from corpus_medline_retrieve.py

    Created on July 18, 2016
    Updated on August 9, 2016
    @author: Wei Wei
'''

import os, cPickle, time, re
from collections import defaultdict, Counter
from knn_data import Data
import numpy as np

class KNN_MeSH(object):
    '''Collect MeSH terms of K nearest neighbors of PMIDs in Lu's data, including all three sets. 
       Compare the MeSH terms with Lu data. H0 is there is no significant change even after this many years,
       and KNN articles have been changed. 
    '''
    def __init__(self, data_dir, knn_dir, mesh_dir, analysis_dir, knn_dict_file):
        self.dataDir = data_dir
        self.knnDir = knn_dir
        self.meshDir = mesh_dir
        self.analyDir = analysis_dir
        self.knn_data = cPickle.load(file(knn_dict_file))

        self.pmid_mesh_dict = defaultdict() ## save {pmid:[mesh]} for all PMIDs, including query and knn pmids
        self.L1000_query_knn_pmid_dict = defaultdict() ## {query pmid: [knn pmid]} in L1000
        self.N2007_query_knn_pmid_dict = defaultdict() ## {query pmid: [knn pmid]} in NLM2007
        self.S200_query_knn_pmid_dict = defaultdict() ## {query pmid: [knn pmid]} in SMALL200

        self.lu_pmid_mesh_dict = defaultdict() ## save {pmid:[mesh]} for all PMIDs, knn pmid from Lu's data
        self.lu_L1000_query_knn_pmid_dict = defaultdict() ## Lu data {query pmid: [knn pmid]} in L1000
        self.lu_N2007_query_knn_pmid_dict = defaultdict() ## Lu data {query pmid: [knn pmid]} in NLM2007
        self.lu_S200_query_knn_pmid_dict = defaultdict()  ## Lu data {query pmid: [knn pmid]} in SMALL200

    def run(self):
        self.load_lu_mesh() ## load KNN pmid from Lu data
        self.load_mesh()
        self.comp_with_lu_data()
        # self.comp_overlap_per_query_pmid_using_MEDLINE()

    def _load_mesh(self, mesh_file):
        terms = file(mesh_file).read().split("\n")
        return terms

    def load_mesh(self):
        '''prepare MeSH terms of given PMIDs'''
        try:
            print "Load BM25 KNN MeSH candidates"
            self.pmid_mesh_dict = cPickle.load(file(os.path.join(self.analyDir,"pmid_mesh_dict.pkl")))
            self.L1000_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"L1000_query_BM25knn_pmid_dict.pkl")))
            self.N2007_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"N2007_query_BM25knn_pmid_dict.pkl")))
            self.S200_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"S200_query_BM25knn_pmid_dict.pkl")))
        except:
            print "Generating BM25 KNN MeSH candidates..."
            query_knn_pmid_dict_list = [self.L1000_query_knn_pmid_dict, self.N2007_query_knn_pmid_dict, self.S200_query_knn_pmid_dict]

            for dt, val in self.knn_data.iteritems():
                print "dataset name: ", dt
                if dt=='L1000':
                    idx = 0
                if dt=='NLM2007':
                    idx = 1
                if dt=='SMALL200':
                    idx = 2

                # pmid_mesh_dict = defaultdict()
                for query_pmid, knn_pmids in val.iteritems():
                    query_knn_pmid_dict_list[idx][query_pmid] = knn_pmids

            ## get all PMIDs in Lu data
            all_pmids = []
            for pmid_dict in query_knn_pmid_dict_list:
                all_pmids+=pmid_dict.keys()
                all_pmids+=[item for sublist in pmid_dict.values() for item in sublist]
            all_pmids = list(set(all_pmids))
            print "total pmid num: ", len(all_pmids)

            for pmid in all_pmids:
                if self.pmid_mesh_dict.get(pmid)==None:            
                    try:
                        pmid_file = os.path.join(self.meshDir, '%s.txt'%pmid)
                        pmid_mesh = self._load_mesh(pmid_file)
                        pmid_mesh = [mesh.lower() for mesh in pmid_mesh]
                        self.pmid_mesh_dict[pmid] = filter(None, pmid_mesh)
                    except Exception as e:
                        pass
            cPickle.dump(self.pmid_mesh_dict, file(os.path.join(self.analyDir,"pmid_mesh_dict.pkl"),"w"))
            cPickle.dump(self.L1000_query_knn_pmid_dict, file(os.path.join(self.analyDir,"L1000_query_BM25knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.N2007_query_knn_pmid_dict, file(os.path.join(self.analyDir,"N2007_query_BM25knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.S200_query_knn_pmid_dict, file(os.path.join(self.analyDir,"S200_query_BM25knn_pmid_dict.pkl"),"w"))

    def mesh_parser4Lu_data(self, text):
        text = re.sub("[-*&]"," ",text)
        mhList = text.split("!")
        mhList = [mh.strip("*") for mh in mhList]
        return mhList

    def load_lu_mesh(self):
        '''load knn mesh terms from Lu data'''
        try:
            print "Load PRC KNN MeSH candidates"
            self.lu_L1000_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"lu_L1000_query_knn_pmid_dict.pkl")))
            self.lu_N2007_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"lu_N2007_query_knn_pmid_dict.pkl")))
            self.lu_S200_query_knn_pmid_dict = cPickle.load(file(os.path.join(self.analyDir,"lu_S200_query_knn_pmid_dict.pkl")))
            self.lu_pmid_mesh_dict = cPickle.load(file(os.path.join(self.analyDir,"lu_pmid_mesh_dict.pkl")))
        except:
            print "Generating PRC KNN MeSH candidates..."
            raw_data_dir = os.path.join(self.dataDir,"lu_data")
            clean_dir = os.path.join(self.dataDir, "lu_data", "clean")

            L1000 = Data(raw_data_dir, clean_dir)
            L1000.large1000()
            NLM2007 = Data(raw_data_dir, clean_dir)
            NLM2007.nlm2007()
            SMALL200 = Data(raw_data_dir, clean_dir)
            SMALL200.small200()
            data_list = [L1000, NLM2007, SMALL200]

            L1000_pmids = L1000.query_pmids
            N2007_pmids = NLM2007.query_pmids
            S200_pmids = SMALL200.query_pmids
            pmid_list = [L1000_pmids, N2007_pmids, S200_pmids]

            query_knn_pmid_list = [self.lu_L1000_query_knn_pmid_dict, self.lu_N2007_query_knn_pmid_dict, self.lu_S200_query_knn_pmid_dict]

            for i in range(len(pmid_list)):
                pmids = pmid_list[i]
                for pmid in pmids:
                    if self.lu_pmid_mesh_dict.get(pmid)==None:
                        try:
                            query_pmid_mesh = self.mesh_parser4Lu_data(data_list[i].query_tam[pmid][2])
                            self.lu_pmid_mesh_dict[pmid]=query_pmid_mesh
                        except:
                            self.lu_pmid_mesh_dict[pmid]=[]

                    knn_pmids = [x[0] for x in data_list[i].nbr_dict[pmid]]
                    query_knn_pmid_list[i][pmid] = knn_pmids

                    for k_pmid in knn_pmids:
                        if self.lu_pmid_mesh_dict.get(k_pmid)==None:
                            try:
                                knn_pmid_mesh = self.mesh_parser4Lu_data(data_list[i].nbr_tam[k_pmid][-1])
                                self.lu_pmid_mesh_dict[k_pmid] = knn_pmid_mesh
                            except:
                                self.lu_pmid_mesh_dict[k_pmid] = []
            cPickle.dump(self.lu_L1000_query_knn_pmid_dict, file(os.path.join(self.analyDir,"lu_L1000_query_knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.lu_N2007_query_knn_pmid_dict, file(os.path.join(self.analyDir,"lu_N2007_query_knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.lu_S200_query_knn_pmid_dict, file(os.path.join(self.analyDir,"lu_S200_query_knn_pmid_dict.pkl"),"w"))
            cPickle.dump(self.lu_pmid_mesh_dict, file(os.path.join(self.analyDir,"lu_pmid_mesh_dict.pkl"),"w"))

    def comp_with_lu_data(self):
        '''compare the mesh terms '''
        ## in every dataset, compare the overlap of MeSH terms in every paper, use the ratio
        self.comp_overlap_per_query_pmid()
        ## compare the count/percentage of individual mesh term
        self.comp_mesh_trends()

    def comp_mesh_trends(self):
        '''Compare the frequencies of individual mesh terms'''
        ## L1000
        L1000_query_pmids = self.L1000_query_knn_pmid_dict.keys()
        latest_knn_mesh_list = []
        lu_knn_mesh_list = []

        for pmid in L1000_query_pmids:
            ## knn mesh terms from latest medline
            knn_pmids = self.L1000_query_knn_pmid_dict.get(pmid)
            for k_pmid in knn_pmids:
                latest_knn_mesh_list += self.pmid_mesh_dict.get(k_pmid)
            ## knn mesh terms from lu data
            lu_knn_pmids = self.lu_L1000_query_knn_pmid_dict.get(pmid)
            for k_pmid in lu_knn_pmids:
                lu_knn_mesh_list += self.lu_pmid_mesh_dict.get(k_pmid)
        latest_knn_mesh_dist = Counter(latest_knn_mesh_list)
        lu_knn_mesh_dist = Counter(lu_knn_mesh_list)

        print "latest_knn_mesh_dist: ", len(latest_knn_mesh_dist)
        print "lu_knn_mesh_dist: ", len(lu_knn_mesh_dist)
        common_terms = set(latest_knn_mesh_dist.keys())&set(lu_knn_mesh_dist.keys())
        print "common terms: ", len(common_terms)
        
    def comp_overlap_per_query_pmid(self):
        '''given a pmid, compare its mesh from lu data and latest 3M'''
        ## L1000
        L1000_query_pmids = self.L1000_query_knn_pmid_dict.keys()
        latest_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        lu_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        for pmid in L1000_query_pmids:
            ## knn mesh terms from latest medline
            knn_pmids = self.L1000_query_knn_pmid_dict.get(pmid)
            local_knn_mesh = []
            for k_pmid in knn_pmids:
                local_knn_mesh += self.pmid_mesh_dict.get(k_pmid)
            latest_query_pmid_knn_mesh_dict[pmid]=Counter(local_knn_mesh)
            ## knn mesh terms from lu data
            lu_knn_pmids = self.lu_L1000_query_knn_pmid_dict.get(pmid)
            lu_local_knn_mesh = []
            for k_pmid in lu_knn_pmids:
                lu_local_knn_mesh += self.lu_pmid_mesh_dict.get(k_pmid)
            lu_query_pmid_knn_mesh_dict[pmid]=Counter(lu_local_knn_mesh)
        ## compare latest_query_pmid_knn_mesh_dict and lu_query_pmid_knn_mesh_dict
        print "L1000"
        print "latest_query_pmid_knn_mesh_dict: ", len(latest_query_pmid_knn_mesh_dict)
        print "lu_query_pmid_knn_mesh_dict: ", len(lu_query_pmid_knn_mesh_dict)
        assert set(latest_query_pmid_knn_mesh_dict.keys())==set(lu_query_pmid_knn_mesh_dict)
        ### compare the occurrence, whether the same mesh term appears in both knn mesh list
        comp_occur = []
        coverage_list = []
        for pmid in L1000_query_pmids:
            std_mesh = self.lu_pmid_mesh_dict.get(pmid)
            latest = set(latest_query_pmid_knn_mesh_dict.get(pmid).keys())
            lu = set(lu_query_pmid_knn_mesh_dict.get(pmid).keys())
            overlap = latest&lu
            comp_occur.append(len(latest&lu)*1.0/len(lu))
            coverage_list.append(len(set(lu)&set(std_mesh))*1.0/len(std_mesh))
        print "comp occurrence"
        print len(comp_occur)
        print np.mean(comp_occur)
        print "L1000 average coverage ", np.mean(coverage_list)

        ## NLM2007
        N2007_query_pmids = self.N2007_query_knn_pmid_dict.keys()
        latest_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        lu_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        for pmid in N2007_query_pmids:
            ## knn mesh terms from latest medline
            knn_pmids = self.N2007_query_knn_pmid_dict.get(pmid)
            local_knn_mesh = []
            for k_pmid in knn_pmids:
                local_knn_mesh += self.pmid_mesh_dict.get(k_pmid)
            latest_query_pmid_knn_mesh_dict[pmid]=Counter(local_knn_mesh)
            ## knn mesh terms from lu data
            lu_knn_pmids = self.lu_N2007_query_knn_pmid_dict.get(pmid)
            lu_local_knn_mesh = []
            for k_pmid in lu_knn_pmids:
                lu_local_knn_mesh += self.lu_pmid_mesh_dict.get(k_pmid)
            lu_query_pmid_knn_mesh_dict[pmid]=Counter(lu_local_knn_mesh)
        ## compare latest_query_pmid_knn_mesh_dict and lu_query_pmid_knn_mesh_dict
        print "NLM2007"
        print "latest_query_pmid_knn_mesh_dict: ", len(latest_query_pmid_knn_mesh_dict)
        print "lu_query_pmid_knn_mesh_dict: ", len(lu_query_pmid_knn_mesh_dict)
        assert set(latest_query_pmid_knn_mesh_dict.keys())==set(lu_query_pmid_knn_mesh_dict)
        ### compare the occurrence, whether the same mesh term appears in both knn mesh list
        coverage_list = []
        comp_occur = []
        for pmid in N2007_query_pmids:
            std_mesh = self.lu_pmid_mesh_dict.get(pmid)
            latest = set(latest_query_pmid_knn_mesh_dict.get(pmid).keys())
            lu = set(lu_query_pmid_knn_mesh_dict.get(pmid).keys())
            overlap = latest&lu
            comp_occur.append(len(latest&lu)*1.0/len(lu))
            coverage_list.append(len(set(lu)&set(std_mesh))*1.0/len(std_mesh))
        print "comp occurrence"
        print len(comp_occur)
        print np.mean(comp_occur)
        print "NLM2007 average coverage ", np.mean(coverage_list)

        ## SMALL200
        S200_query_pmids = self.S200_query_knn_pmid_dict.keys()
        latest_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        lu_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        for pmid in S200_query_pmids:
            ## knn mesh terms from latest medline
            knn_pmids = self.S200_query_knn_pmid_dict.get(pmid)
            local_knn_mesh = []
            for k_pmid in knn_pmids:
                local_knn_mesh += self.pmid_mesh_dict.get(k_pmid)
            latest_query_pmid_knn_mesh_dict[pmid]=Counter(local_knn_mesh)
            ## knn mesh terms from lu data
            lu_knn_pmids = self.lu_S200_query_knn_pmid_dict.get(pmid)
            lu_local_knn_mesh = []
            for k_pmid in lu_knn_pmids:
                lu_local_knn_mesh += self.lu_pmid_mesh_dict.get(k_pmid)
            lu_query_pmid_knn_mesh_dict[pmid]=Counter(lu_local_knn_mesh)
        ## compare latest_query_pmid_knn_mesh_dict and lu_query_pmid_knn_mesh_dict
        print "SMALL200"
        print "latest_query_pmid_knn_mesh_dict: ", len(latest_query_pmid_knn_mesh_dict)
        print "lu_query_pmid_knn_mesh_dict: ", len(lu_query_pmid_knn_mesh_dict)
        assert set(latest_query_pmid_knn_mesh_dict.keys())==set(lu_query_pmid_knn_mesh_dict)
        ### compare the occurrence, whether the same mesh term appears in both knn mesh list
        coverage_list = []
        comp_occur = []
        for pmid in S200_query_pmids:
            std_mesh = self.lu_pmid_mesh_dict.get(pmid)
            latest = set(latest_query_pmid_knn_mesh_dict.get(pmid).keys())
            lu = set(lu_query_pmid_knn_mesh_dict.get(pmid).keys())
            overlap = latest&lu
            comp_occur.append(len(latest&lu)*1.0/len(lu))
            coverage_list.append(len(set(lu)&set(std_mesh))*1.0/len(std_mesh))
        print "comp occurrence"
        print len(comp_occur)
        print np.mean(comp_occur)
        print "SMALL200 average coverage ", np.mean(coverage_list)    

    def comp_overlap_per_query_pmid_using_MEDLINE(self):
        '''Use downloaded MEDLINE data, instead of pmid-mesh pairs from Lu's data'''
        
        mesh_1997_dir = os.path.join(self.analyDir, "mesh_1995_1997") 
        print "mesh_1997_dir: ", mesh_1997_dir
        N2007_query_pmids = self.N2007_query_knn_pmid_dict.keys()
        latest_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        lu_query_pmid_knn_mesh_dict = defaultdict() ## {query pmid: [knn mesh Counter object]}
        for pmid in N2007_query_pmids:
            ## knn mesh terms from latest medline
            knn_pmids = self.N2007_query_knn_pmid_dict.get(pmid)
            local_knn_mesh = []
            for k_pmid in knn_pmids:
                try:
                    k_mesh = file(os.path.join(mesh_1997_dir,"%s.txt"%k_pmid)).read().split('\n')
                    print k_mesh
                    local_knn_mesh+=k_mesh
                except Exception as e:
                    print e
                    # local_knn_mesh += self.pmid_mesh_dict.get(k_pmid)
                print k_pmid
                print local_knn_mesh
                raw_input('...')
            latest_query_pmid_knn_mesh_dict[pmid]=Counter(local_knn_mesh)
            ## knn mesh terms from lu data
            lu_knn_pmids = self.lu_N2007_query_knn_pmid_dict.get(pmid)
            lu_local_knn_mesh = []
            for k_pmid in lu_knn_pmids:
                lu_local_knn_mesh += self.lu_pmid_mesh_dict.get(k_pmid)
            lu_query_pmid_knn_mesh_dict[pmid]=Counter(lu_local_knn_mesh)
        ## compare latest_query_pmid_knn_mesh_dict and lu_query_pmid_knn_mesh_dict
        print "NLM2007"
        print "latest_query_pmid_knn_mesh_dict: ", len(latest_query_pmid_knn_mesh_dict)
        print "lu_query_pmid_knn_mesh_dict: ", len(lu_query_pmid_knn_mesh_dict)
        assert set(latest_query_pmid_knn_mesh_dict.keys())==set(lu_query_pmid_knn_mesh_dict)
        ### compare the occurrence, whether the same mesh term appears in both knn mesh list
        coverage_list = []
        comp_occur = []
        for pmid in N2007_query_pmids:
            std_mesh = self.lu_pmid_mesh_dict.get(pmid)
            latest = set(latest_query_pmid_knn_mesh_dict.get(pmid).keys())
            lu = set(lu_query_pmid_knn_mesh_dict.get(pmid).keys())
            overlap = latest&lu
            comp_occur.append(len(latest&lu)*1.0/len(lu))
            coverage_list.append(len(set(lu)&set(std_mesh))*1.0/len(std_mesh))
        print "comp occurrence"
        print len(comp_occur)
        print np.mean(comp_occur)
        print "NLM2007 average coverage ", np.mean(coverage_list)        

if __name__ == "__main__":
    time_span = "latest_1M"
    data_dir = '/home/w2wei/data'
    knn_base_dir = os.path.join(data_dir, "knn")
    knn_dir = os.path.join(knn_base_dir, "knn_%s"%time_span)
    if not os.path.exists(knn_dir):
        os.makedirs(knn_dir)

    lu_data_knn_on_latest_1M_file = os.path.join(knn_dir, "lu_data_BM25_KNN.pkl")## BM25 KNN from latest 1M MEDLINE. \
                                                                                 ## generated from candidates_retrieve.py, original experiment, commented.
    mesh_base_dir = os.path.join(data_dir, "mesh") ## mesh terms generated using candidates_index.py
    mesh_dir = os.path.join(mesh_base_dir, "mesh_%s"%time_span)

    analysis_base_dir = os.path.join(data_dir, "analysis")
    analysis_dir = os.path.join(analysis_base_dir, "candidates_BM25_KNN_on_MEDLINE_%s"%time_span)
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    knn_mesh = KNN_MeSH(data_dir, knn_dir, mesh_dir, analysis_dir, lu_data_knn_on_latest_1M_file)
    knn_mesh.run()


    # data_dir = '/home/w2wei/data'
    # knn_dir = os.path.join(data_dir, "knn_from_clean")
    # knn_3M_file = os.path.join(knn_dir, "lu_data_knn_from_3M.pkl") ## BM25 KNN from latest 3M MEDLINE. generated from corpus_medline_retrieve.py, original experiment, commented.
    # latest_3M_mesh_dir = os.path.join(data_dir, 'latest_3M_mesh')
    # analysis_dir = os.path.join(knn_dir, "analysis_lu_data_3M")
    # if not os.path.exists(analysis_dir):
    #     os.makedirs(analysis_dir)
    # knn_mesh = KNN_MeSH(data_dir, knn_dir, knn_3M_file, latest_3M_mesh_dir, analysis_dir)
    # knn_mesh.run()