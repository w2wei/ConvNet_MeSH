'''
    Find the time span of Huang's datasets, Small200, NLM2007, L1000

    Created on August 25, 2016
    Updated on August 25, 2016
    @author: Wei Wei
'''

import os, re
from collections import Counter
from Bio import Entrez
Entrez.email = "granitedewint@gmail.com"

class GetMEDLINE(object):

    def __init__(self, pmid_file, medline_file):
        self.pmidList = filter(None, file(pmid_file).read().split("\n"))
        self.medlineFile = medline_file

    def get_medline(self):
        query = ",".join(self.pmidList)
        try:
            handle=Entrez.efetch(db="pubmed",id=query,rettype="medline",retmode="text")
            record=handle.read()
            handle.close()
            # output
            fout=file(self.medlineFile,"w")
            fout.write(record)
        except Exception as e:
            print "error msg: ", e

def collect_data(pmid_file, medline_file):
    if not os.path.exists(medline_file):
        q = GetMEDLINE(pmid_file, medline_file)
        q.get_medline()

def analyze_medline(medline_file):
    '''extract publish dates of collected documents and get a distribution'''
    yearList = []
    for line in file(medline_file):
        if line.startswith("DP  -"):
            year = re.search("[0-9]{4}", line).group(0)
            yearList.append(year)
    yearDist = Counter(yearList)
    years = yearDist.keys()
    years.sort()
    for year in years:
        print year, yearDist.get(year)

if __name__=="__main__":
    ## load Hunag's datasets
    data_dir = "/home/w2wei/data/lu_data"
    S200_dir = os.path.join(data_dir, "SMALL200")
    N2007_dir= os.path.join(data_dir, "NLM2007")
    L1000_dir = os.path.join(data_dir, "L1000")

    S200_pmid_file = os.path.join(S200_dir, "S200.pmids")
    N2007_pmid_file = os.path.join(N2007_dir, "NLM2007.pmids")
    L1000_pmid_file = os.path.join(L1000_dir, "L1000.pmids")

    S200_MEDLINE_file = os.path.join(S200_dir, "S200.medline")
    N2007_MEDLINE_file = os.path.join(N2007_dir, "NLM2007.medline")
    L1000_MEDLINE_file = os.path.join(L1000_dir, "L1000.medline")

    ## For L1000
    collect_data(L1000_pmid_file, L1000_MEDLINE_file)
    analyze_medline(L1000_MEDLINE_file)






