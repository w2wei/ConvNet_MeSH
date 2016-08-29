'''
    Load MEDLINE records and extract titles, abstracts, and MeSH terms. 
    Return the results in dictionaries, {PMID:[[title],[abstract]]}, {PMID:[MeSH terms]}
    Prepare data for Point-wise LTR algorithm.
    Created on Jan 17, 2016,
    Updated on Jun 1, 2016

    @author: Wei Wei
'''

import os, pickle, time
# from Utilities import Analyzer

class Corpus(object):
    '''Parse raw MEDLINE records; extract PMID, title, abstract, and MeSH'''
    def __init__(self, medline_dir, medline_file, outDataDir):
        self.medDir = medline_dir
        self.outDataDir = outDataDir
        self.inFile = os.path.join(self.medDir, medline_file)
        self.tiab = {}
        self.mesh = {}

    def run(self):
        self.tiab = self.extractTiAb(self.inFile)
        self.mesh = self.extractMH(self.inFile)
        # self.saveTiAb(tiab)  # write files to disk

    def saveTiAb(self, TiAbDict):
        for pmid in TiAbDict.keys():
            if pmid in os.listdir(self.outDataDir):
                continue
            titleStr = (TiAbDict[pmid][0].strip(".") + ". ")
            absStr = TiAbDict[pmid][1].strip(".") + ". "
            text = " ".join(Analyzer().run(titleStr + absStr)) # tokenized and stemmed text, lower case and no stop words
            ftext = file(os.path.join(self.outDataDir, pmid), "w")
            ftext.write(text) 

    def extractTiAb(self, filePath):
        '''Extract the title, abstract from a single MEDLINE file'''
        medlineRecs = file(filePath).readlines()
        totalLine = len(medlineRecs)
        newEntry = False
        firstLine = True
        newField = True
        ti = False
        title = []
        ab = False
        abstract = []
        textDict = {}
        lineCount = 0
        pmid = ''
        for line in medlineRecs:
            lineCount += 1
            if line.startswith("PMID-"):
                pmid = line.split("- ")[1].strip("\n")  # \n\r
                newEntry = True
                firstLine = False
                newField = True
                abstract = []
                ab = False
                title = []
                ti = False

            if newEntry and line.startswith("AB  - "):
                absRaw = line.split("AB  - ")[1].strip("\n")  # \n\r
                abstract.append(absRaw.strip(" "))
                newField = False
                ab = True

            if newEntry and line.startswith("TI  - "):
                ttRaw = line.split("TI  - ")[1].strip("\n").strip("[").strip(".").strip("]")
                title.append(ttRaw)
                newField = False
                ti = True

            if line.startswith(" "):
                if not newField and ab:
                    abstract.append(line.strip("\n").strip(" "))
                if not newField and ti:
                    title.append(line.strip("\n").strip(" "))

            if not line.startswith("AB") and not line.startswith("TI") and not line.startswith(" "):
                newField = True
                ti = False
                ab = False

            if (not firstLine) and line == "\n":
                textDict[pmid] = [" ".join(title), " ".join(abstract)]
                newEntry = False

            if lineCount == totalLine:  # check if this is the end of the file
                textDict[pmid] = [" ".join(title), " ".join(abstract)]
                newEntry = False

        return textDict

    def extractMH(self, filePath):
        '''Extract MeSH terms from a single MEDLINE file'''
        medlineRecs = file(filePath).readlines()
        totalLine = len(medlineRecs)
        newEntry = False
        firstLine = True
        mhList = []        
        textDict = {}
        lineCount = 0
        for line in medlineRecs:
            lineCount += 1
            if line.startswith("PMID-"):
                pmid = line.split("- ")[1].strip("\n")  # \n\r
                newEntry = True
                firstLine = False
                mhList = []
                
            if newEntry and line.startswith("MH  - "):
                mhRaw = line.split("MH  - ")[1].strip("\n").lower()  # \n\r
                mhRaw = filter(None, mhRaw.split("/"))[:1]# only the main heading
                # if len(mhRaw)>1: # qualifier exists
                #     mhRaw = mhRaw[:1]+mhRaw
                mhList += mhRaw
                mhList = [item.strip("*") for item in mhList]
        
            if (not firstLine) and line == "\n":
                textDict[pmid] = mhList
                newEntry = False
            
            if lineCount == totalLine:  # check if this is the end of the file
                textDict[pmid] = mhList
                newEntry = False     
        return textDict


def loadCorpus(rawDataFile):
    ## for idash-meta-dev
    # data_dir = "/home/w2wei/Research/mesh/data/deep_pmcoa/pointwise_ltr/sample"
    # rawdata_dir = os.path.join(data_dir, "raw") ## raw data
    # outdata_dir = os.path.join(data_dir, "clean") ## processed data
    ## for idash-cloud
    data_dir = "/home/w2wei/projects/pointwiseLTR/data"
    rawdata_dir = os.path.join(data_dir, "sample")
    outdata_dir = os.path.join(rawdata_dir, "clean_data") ## processed data
    for path in [data_dir, rawdata_dir, outdata_dir]:
        if not os.path.exists(path):
            os.makedirs(path)
    corpus = Corpus(rawdata_dir, rawDataFile, outdata_dir)
    corpus.run()
    print "corpus ready"
    return corpus

if __name__=="__main__":
    loadRawData()
