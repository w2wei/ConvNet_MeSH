'''
    Retrieve MEDLINE associated with PMIDs in Small200, NLM2007, and L1000

    Created on August 27, 2016
    Updated on August 27, 2016

    @author: Wei Wei
'''

from Bio import Entrez
from datetime import datetime
import os,time,sys
from multiprocessing import Pool, cpu_count

Entrez.email = "granitedewint@gmail.com"

    
class GetMEDLINE(object):
    '''Query PubMed for recent PMID and then collect MEDLINE'''

    def __init__(self, pmid_file_dir, medline_file_dir, year):
        self.pmidDir = os.path.join(pmid_file_dir, str(year))
        self.medlineDir = os.path.join(medline_file_dir, str(year))
        self.query = """("%d/01/01"[Publication Date] : "%d/12/31"[Publication Date])"""%(year, year)
        self.year = year

    def get_PMID(self):
        """Retrieve PMID"""
        if not os.path.exists(self.pmidDir):
            os.makedirs(self.pmidDir)

        retstart = 0
        step = 5000 # restricted by NCBI
        i=0
        newPMID=step
        count=0

        while newPMID>=step:
            try:
                handle=Entrez.esearch(db='pubmed',term=self.query,retstart=i*step,retmax=step)
                record=Entrez.read(handle)
                newPMID=len(record["IdList"])
                count+=newPMID
                i+=1
                fname = "%s_%d.txt"%(self.year, i)
                fout = file(os.path.join(self.pmidDir, fname),"w")
                fout.write("\n".join(record["IdList"]))
                if count%1000 == 0:
                    print self.year, " PMID count: ", count
            except Exception as e:
                print e
                print "PMID error year: ", self.year

        print "Total PMID count: ",count

    def get_medline(self):
        if not os.path.exists(self.medlineDir):
            os.makedirs(self.medlineDir)

        pmidFiles = os.listdir(self.pmidDir)
        count = 0
        for pmidDoc in pmidFiles:
            pmidList = file(os.path.join(self.pmidDir, pmidDoc)).read().split("\n")
            # print "PMID num: ", len(pmidList)
            query = ",".join(pmidList)
            try:
                handle=Entrez.efetch(db="pubmed",id=query,rettype="medline",retmode="text")
                record=handle.read()
                handle.close()
                # output
                fout=file(os.path.join(self.medlineDir, pmidDoc),"w")
                fout.write(record)
            except Exception as e:
                print "PMID doc: ", pmidDoc
                print "error msg: ", e
            count+=1
            if count%1000==0:
                print self.year, " MEDLINE count: ", count

def collect_pmid(year):
    data_dir = "/home/w2wei/data/"
    pmid_file_base_dir = os.path.join(data_dir, "pmid_docs_by_year")
    medline_file_base_dir = os.path.join(data_dir, "medline_docs_by_year")
    q = GetMEDLINE(pmid_file_base_dir, medline_file_base_dir, year)
    q.get_PMID()

def collect_medline(year):
    data_dir = "/home/w2wei/data/"
    pmid_file_base_dir = os.path.join(data_dir, "pmid_docs_by_year")
    medline_file_base_dir = os.path.join(data_dir, "medline_docs_by_year")
    q = GetMEDLINE(pmid_file_base_dir, medline_file_base_dir, year)
    q.get_medline()    

if __name__=="__main__":
    time_span = sys.argv[1]
    print "Collecting PMIDs and MEDLINE during %s..."%(time_span)
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    years = range(int(startyear),int(endyear)+1)

    pool = Pool(cpu_count())
    pool.map(collect_pmid, years)

    pool = Pool(cpu_count())
    pool.map(collect_medline, years)

