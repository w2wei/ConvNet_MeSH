'''
This script prepares data for the algorithm reported in Huang M. et.al. Recommending MeSH terms for annotating biomedical articles. JAMIA. 2011
    Created on Oct 13, 2015
    Updated on Feb 25, 2016
    @author: Wei Wei
'''

import os, pickle, re

class Data(object):
    '''This class loads training/test data sets and MeSH terms
       Validated on NLM2007
    '''
    def __init__(self,dataDir, wkDir):
        self.dataDir = dataDir
        self.wkDir = wkDir

        self.query_pmids = []
        self.nbr_pmids = []
        self.query_tam = {}
        self.nbr_tam = {}
        self.nbr_dict = {}

    def loadPMID(self, dir, outFile):
        text = file(dir).read().split("\n")
        text = filter(None,text)
        pickle.dump(text,file(outFile,"w"))
        return text

    def loadContents(self, dir, outFile):
        text = file(dir).read().split("\n")
        text = filter(None, text)
        pmid_tam_dict = {}
        for item in text:
            result = item.split("|")
            pmid = result[0]
            ctg = result[1]
            content = result[2:]
            if pmid not in pmid_tam_dict.keys():
                pmid_tam_dict[pmid]=[]
            if ctg == "t" or ctg == "a":
                pmid_tam_dict[pmid].append(content)
            if ctg == "m":
                meshList = []
                rl1 =[]
                rl2 =[]
                for mh in content:
                    meshList.append(re.split("[!*]",mh)[0]) # ! plays the role of /, if * and ! do not co-exist
                meshList = filter(None, meshList)
                meshList = list(set(meshList))
                pmid_tam_dict[pmid].append("!".join(meshList))
        pickle.dump(pmid_tam_dict,file(outFile,"w"))
        return pmid_tam_dict

    def loadRelationScore(self,dir,outFile):
        result = {}
        raw = filter(None, file(dir).read().split("\n"))
        for item in raw:
            try:
                query, nbr, score = item.split("\t")
            except Exception as e:
                print "Data.loadRelationScore error"
                print query
                print 
                raw_input("wait...")
            if query not in result.keys():
                result[query] = [(nbr, float(score))]
            else:
                result[query].append((nbr,float(score)))
        pickle.dump(result, file(outFile,"w"))
        return result

    def small200(self):
        small_dir = os.path.join(self.dataDir,"SMALL200")
        query_pmid_file = os.path.join(small_dir,"S200.pmids")
        query_mh_file = os.path.join(small_dir,"S200.TiAbMe")
        # nbr_pmid_file = os.path.join(small_dir,"S200_50neighbors.pmids")
        nbr_mh_file = os.path.join(small_dir,"S200_50neighbors.TiAbMe")
        query_nbr_dict_file = os.path.join(small_dir,"S200_50neighbors.score")
        small200_wkdir = os.path.join(self.wkDir, "SMALL200")
        if not os.path.exists(small200_wkdir):
            os.makedirs(small200_wkdir)
        query_pmid_outfile = os.path.join(small200_wkdir,"query_pmids.pkl")
        # nbr_pmid_outfile = os.path.join(small200_wkdir,"nbr_pmids.pkl")
        query_tam_outfile = os.path.join(small200_wkdir,"query_tam.pkl")
        nbr_tam_outfile = os.path.join(small200_wkdir, "nbr_tam.pkl")
        query_nbr_dict_outfile = os.path.join(small200_wkdir, "nbr_dict.pkl")
        try:
            self.query_pmids = pickle.load(file(query_pmid_outfile))
        except:
            self.query_pmids = self.loadPMID(query_pmid_file, query_pmid_outfile)
        try:
            self.query_tam = pickle.load(file(query_tam_outfile))
        except:
            self.query_tam = self.loadContents(query_mh_file, query_tam_outfile) # {pmid:[title, abstract, mesh]}
        try:
            self.nbr_tam = pickle.load(file(nbr_tam_outfile))
        except:
            self.nbr_tam = self.loadContents(nbr_mh_file, nbr_tam_outfile)

        try:
            self.nbr_dict = pickle.load(file(query_nbr_dict_outfile))
        except:
            self.nbr_dict = self.loadRelationScore(query_nbr_dict_file, query_nbr_dict_outfile)

    def nlm2007(self):
        nlm2007_dir = os.path.join(self.dataDir,"NLM2007")
        query_pmid_file = os.path.join(nlm2007_dir,"NLM2007.pmids")
        query_mh_file = os.path.join(nlm2007_dir,"NLM2007.TiAbMe")
        # nbr_pmid_file = os.path.join(nlm2007_dir,"NLM2007_50neighbors.pmids")
        nbr_mh_file = os.path.join(nlm2007_dir,"NLM2007_50neighbors.TiAbMe")
        query_nbr_dict_file = os.path.join(nlm2007_dir,"NLM2007_50neighbors.score")
        nlm2007_wkdir = os.path.join(self.wkDir, "NLM2007")
        if not os.path.exists(nlm2007_wkdir):
            os.makedirs(nlm2007_wkdir)
        query_pmid_outfile = os.path.join(nlm2007_wkdir,"query_pmids.pkl")
        # nbr_pmid_outfile = os.path.join(nlm2007_wkdir,"nbr_pmids.pkl")
        query_tam_outfile = os.path.join(nlm2007_wkdir,"query_tam.pkl")
        nbr_tam_outfile = os.path.join(nlm2007_wkdir, "nbr_tam.pkl")
        query_nbr_dict_outfile = os.path.join(nlm2007_wkdir, "nbr_dict.pkl")
        try: ## self.query_pmids validated on NLM2007
            self.query_pmids = pickle.load(file(query_pmid_outfile))
        except:
            self.query_pmids = self.loadPMID(query_pmid_file, query_pmid_outfile)
        try: ## self.query_tam validated on NLM2007
            self.query_tam = pickle.load(file(query_tam_outfile))
        except:
            self.query_tam = self.loadContents(query_mh_file, query_tam_outfile) # {pmid:[title, abstract, mesh]}
        try: ## self.nbr_tam validated on NLM2007
            self.nbr_tam = pickle.load(file(nbr_tam_outfile))
        except:
            self.nbr_tam = self.loadContents(nbr_mh_file, nbr_tam_outfile)
        try: ## self.nbr_dict validated on NLM2007
            self.nbr_dict = pickle.load(file(query_nbr_dict_outfile))
        except:
            self.nbr_dict = self.loadRelationScore(query_nbr_dict_file, query_nbr_dict_outfile)

    def large1000(self):
        large1000_dir = os.path.join(self.dataDir,"L1000")
        query_pmid_file = os.path.join(large1000_dir,"L1000.pmids")
        query_mh_file = os.path.join(large1000_dir,"L1000.TiAbMe")
        nbr_mh_file = os.path.join(large1000_dir,"L1000_50neighbors.TiAbMe")
        query_nbr_dict_file = os.path.join(large1000_dir,"L1000_50neighbors.score")
        large1000_wkdir = os.path.join(self.wkDir, "L1000")
        if not os.path.exists(large1000_wkdir):
            os.makedirs(large1000_wkdir)
        query_pmid_outfile = os.path.join(large1000_wkdir,"query_pmids.pkl")
        query_tam_outfile = os.path.join(large1000_wkdir,"query_tam.pkl")
        nbr_tam_outfile = os.path.join(large1000_wkdir, "nbr_tam.pkl")
        query_nbr_dict_outfile = os.path.join(large1000_wkdir, "nbr_dict.pkl")
        try: ## self.query_pmids validated on NLM2007
            self.query_pmids = pickle.load(file(query_pmid_outfile))
        except:
            self.query_pmids = self.loadPMID(query_pmid_file, query_pmid_outfile)
        try: ## self.query_tam validated on NLM2007
            self.query_tam = pickle.load(file(query_tam_outfile))
        except:
            self.query_tam = self.loadContents(query_mh_file, query_tam_outfile) # {pmid:[title, abstract, mesh]}
        try: ## self.nbr_tam validated on NLM2007
            self.nbr_tam = pickle.load(file(nbr_tam_outfile))
        except:
            self.nbr_tam = self.loadContents(nbr_mh_file, nbr_tam_outfile)
        try: ## self.nbr_dict validated on NLM2007
            self.nbr_dict = pickle.load(file(query_nbr_dict_outfile))
        except:
            self.nbr_dict = self.loadRelationScore(query_nbr_dict_file, query_nbr_dict_outfile)

if __name__ == '__main__':
    data = Data(data_dir, base_dir)
    data.nlm2007()
    print "Data ready" 
