'''
	Find the latest 9M MEDLINE from 10M records during 2004 and 2016


	Created on Jul 15, 2016
    Updated on Jul 15, 2016

    @author: Wei Wei
'''

import os, time, re, cPickle
from collections import defaultdict

class Find_latest(object):
    '''Build an index for all pmids and medline files. pmid: medline file name. This class will be deprecated.'''
    def __init__(self, data_dir, result_dir, pmid_file_dir, medline_file_dir):
        self.dataDir = data_dir
        self.resDir = result_dir
        self.pmidDir = pmid_file_dir
        self.medlineDir = medline_file_dir
        self.pmidList = []
        self.latest_med_files = []
        self.pmid_medFile_dict = defaultdict()
        self.latest_pmid_dict = defaultdict()

    def find(self):
        self.sort_pmids()
        self.index_medfiles()
        self.find_latest_9M_medline()
        self.save_9M_medline()

    def sort_pmids(self):
        for doc in os.listdir(self.pmidDir):## load all PMIDs, 10M records, 99MB
            pmidList = file(os.path.join(self.pmidDir, doc)).read().split("\n")
            self.pmidList+=pmidList
        self.pmidList.sort(reverse=True)

    def index_medfiles(self):
        try:
            self.pmid_medFile_dict = cPickle.load(file(os.path.join(self.resDir,"pmid_medFile_dict.pkl")))
        except:
            for doc in os.listdir(self.medlineDir):
                text = file(os.path.join(self.medlineDir, doc)).read().split("\n")
                pmidLines = [line for line in text if line.startswith("PMID- ")]
                pmidList = [line.split("- ")[1] for line in pmidLines]
                [self.pmid_medFile_dict.setdefault(pmid, []).append(doc) for pmid in pmidList]
            cPickle.dump(self.pmid_medFile_dict, file(os.path.join(self.resDir,"pmid_medFile_dict.pkl"),"w"))

    def find_latest_9M_medline(self):
        print "total PMID num: ", len(self.pmidList)
        latest = self.pmidList[1000000:] # 2000000
        print "latest PMID num: ", len(latest)
        
        for pmid in latest:
            try:
                filename = self.pmid_medFile_dict.get(pmid)
                # print pmid, filename
                self.latest_med_files += self.pmid_medFile_dict.get(pmid)
                self.latest_pmid_dict[pmid] = 1
            except Exception as e:
                pass
        self.latest_med_files = list(set(self.latest_med_files))

    def save_9M_medline(self):
        latest_9M_dir = os.path.join(self.dataDir, "latest_9M_medline_docs")
        if not os.path.exists(latest_9M_dir):
            os.makedirs(latest_9M_dir)

        latest_med_docs = defaultdict()
        idx = 0

        med_docs = os.listdir(self.medlineDir)
        for med_doc in self.latest_med_files:
            text = file(os.path.join(self.medlineDir, med_doc)).read()
            med_recs = text.split("\n\n")
            for rec in med_recs:
                try:
                    pmid = rec[:15].split("PMID- ")[1].strip()
                    if self.latest_pmid_dict.get(pmid):
                        latest_med_docs[pmid]=rec
                except Exception as e:
                    print e
                    print rec
                    continue
                if len(latest_med_docs)==5000:
                    out = latest_med_docs.values()
                    fout = file(os.path.join(latest_9M_dir, "%d.txt"%idx),"w")
                    fout.write("\n\n".join(out))
                    latest_med_docs = defaultdict()
                    idx+=1

if __name__=="__main__":
    data_dir = "/home/w2wei/data/"
    result_dir = "/home/w2wei/project/results/"
    pmid_file_dir = os.path.join(data_dir, "pmid_docs")
    medline_file_dir = os.path.join(data_dir, "medline_docs")
    med_index = Find_latest(data_dir, result_dir, pmid_file_dir, medline_file_dir)
    med_index.find()

