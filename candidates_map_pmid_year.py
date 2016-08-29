'''
    Build a dictionary of PMID and years.

    Created on August 28, 2016
    Updated on August 28, 2016
    @author: Wei Wei
'''

import os, cPickle, time
from collections import defaultdict

def build_pmid_year_dict(inDir, outFile):
    try:
        pmid_year_dict = cPickle.load(file(outFile))
    except:
        pmid_year_dict = defaultdict()
        years = os.listdir(inDir)
        print "Avaiable years: ", len(years)
        for year in years:
            docs = os.listdir(os.path.join(inDir, year))
            for doc in docs:
                pmid = doc[:-4]
                pmid_year_dict[pmid]=str(year)
        cPickle.dump(pmid_year_dict, file(outFile,'wb'), protocol=cPickle.HIGHEST_PROTOCOL) 

if __name__=="__main__":
    data2_dir = "/home/w2wei/data2"
    tiab_base_dir = os.path.join(data2_dir, "tiab_by_year")
    util_dir = os.path.join(data2_dir,"utils")
    if not os.path.exists(util_dir):
        os.makedirs(util_dir)
    pmid_year_dict_file = os.path.join(util_dir,"pmid_year_dict.pkl")
    build_pmid_year_dict(tiab_base_dir, pmid_year_dict_file)

