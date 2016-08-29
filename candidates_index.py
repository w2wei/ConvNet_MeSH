'''
    Process the raw MEDLINE docuemnts and extract the parts to be indexed.
    Save one document in one file.

    Index the latest 9M MEDLINE using pyLucene.
	Before running this script, install PyLucene (https://lucene.apache.org/pylucene/install.html).
	1. Download pylucence installer.
	2. Before install JCC, install python-dev. Command: apt-get install python-dev. This will set up c++ Python.h etc. \
	   make sure JCC is correctly installed. Check JAVA_HOME, java version
	3. Install ANT. Check ANT_HOME, etc

	Created on July 15, 2016
	Updated on August 7, 2016
	@author: Wei Wei
'''

import os, re, time, sys
import multiprocessing as mp
from datetime import datetime
import lucene, threading
# from candidates_IndexFiles import IndexFiles
from candidates_IndexFiles_new import IndexFiles
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer
from nltk.corpus import stopwords

from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version

from org.apache.lucene.analysis.core import LowerCaseFilter, StopFilter, StopAnalyzer
from org.apache.lucene.analysis.en import PorterStemFilter
from org.apache.lucene.analysis.standard import StandardTokenizer, StandardFilter
from org.apache.pylucene.analysis import PythonAnalyzer
from org.apache.lucene.search.similarities import BM25Similarity

def extract_MeSH_from_MEDLINE(input_dir, output_dir):
    '''extract MeSH from raw MEDLINE files and save one article in one file'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = mp.JoinableQueue()
    
    num_consumers = mp.cpu_count()
    print "creating %d consumers " % num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    for doc in os.listdir(input_dir):
        inFile = os.path.join(input_dir, doc)
        tasks.put(MeSH_Task(inFile, output_dir))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

def extract_TiAb_from_MEDLINE(input_dir, output_dir):
    '''extract titles and abstracts from raw MEDLINE files and save one article in one file'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = mp.JoinableQueue()
    
    num_consumers = mp.cpu_count()
    print "creating %d consumers " % num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    for doc in os.listdir(input_dir):
        inFile = os.path.join(input_dir, doc)
        tasks.put(Task(inFile, output_dir))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()
    
class Consumer(mp.Process):
    def __init__(self, task_queue):  # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" % mp.current_process()
                self.task_queue.task_done()
                break
            next_task.__call__()
            self.task_queue.task_done()
        return

class MeSH_Task(object):
    def __init__(self, inFile, outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        '''Extract titles and abstracts from MEDLINE'''
        # doc_path = os.path.join(self.inputFile, doc)
        pmid_mesh_dict = self.__extractMeSH(self.inputFile)
        self.__save__(pmid_mesh_dict)

    def __extractMeSH(self, filePath):
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
                mhRaw = line.split("MH  - ")[1].strip("\n")  # \n\r
                mhRaw = filter(None, mhRaw.split("/"))
                if len(mhRaw)>1: # qualifier exists
                    mhRaw = mhRaw[:1]
                mhList += mhRaw
                mhList = [item.strip("*") for item in mhList]
        
            if (not firstLine) and line == "\n":
                textDict[pmid] = mhList
                newEntry = False
            
            if lineCount == totalLine:  # check if this is the end of the file
                textDict[pmid] = mhList
                newEntry = False     
        return textDict

    def __save__(self, pmid_mesh_dict):
        for pmid, content in pmid_mesh_dict.iteritems():
            fout = file(os.path.join(self.outputFile, "%s.txt"%pmid),"w")
            # fout.write(str(content))
            fout.write("\n".join(content))

    def __str__(self):
        return "%s " % (self.inputFile)

class Task(object):
    def __init__(self, inFile, outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        '''Extract titles and abstracts from MEDLINE'''
        # doc_path = os.path.join(self.inputFile, doc)
        pmid_tiab_dict = self.__extractTiAb(self.inputFile)
        # assert len(pmid_tiab_dict)==5000
        self.__save__(pmid_tiab_dict)

    def __extractTiAb(self, filePath):
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
        
    def __save__(self, pmid_tiab_dict):
        for pmid, content in pmid_tiab_dict.iteritems():
            fout = file(os.path.join(self.outputFile, "%s.txt"%pmid),"w")
            content += [content[0]] ## duplicate the title in accordance with PRC
            fout.write("\n".join(content))

    def __str__(self):
        return "%s " % (self.inputFile)

class PorterStemmerAnalyzer(PythonAnalyzer):
    def createComponents(self, fieldName, reader):
        source = StandardTokenizer(Version.LUCENE_CURRENT, reader)
        filter = StandardFilter(Version.LUCENE_CURRENT, source)
        filter = LowerCaseFilter(Version.LUCENE_CURRENT, filter)  # normalize token text to lower case
        filter = PorterStemFilter(filter)  # transform the token stream as per the Porter stemming algorithm
        filter = StopFilter(Version.LUCENE_CURRENT, filter,
                            StopAnalyzer.ENGLISH_STOP_WORDS_SET)
        return self.TokenStreamComponents(source, filter)

class Index(object):
    '''This class indexes prepared MEDLINE documents'''
    def __init__(self, doc_dir, index_dir, index_file):
        self.docDir = doc_dir
        self.indexDir = index_dir
        self.indexFile = index_file
        try:
            env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
            if self.indexFile not in os.listdir(self.indexDir):
                threading.Thread(target=self.worker, args=(env,)).start()
        except Exception as e:
            print e
            print "Documents for %s not indexed!"%index_file

    def worker(self, env):
        env.attachCurrentThread()
        start = datetime.now()
        IndexFiles(self.docDir, os.path.join(self.indexDir, self.indexFile), PorterStemmerAnalyzer())  # PorterStemmer analyzer
        end = datetime.now()
        print end - start

class Index_new(object):
    '''This class indexes prepared MEDLINE documents'''
    
    def __init__(self, doc_dir, index_dir, index_file, year_list):
        self.docDir = doc_dir
        self.indexDir = index_dir
        self.indexFile = index_file
        self.yearList = year_list
        try:
            env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
            if self.indexFile not in os.listdir(self.indexDir):
                threading.Thread(target=self.worker, args=(env,)).start()
        except Exception as e:
            print e
            print "Documents for %s not indexed!"%index_file

    def worker(self, env):
        env.attachCurrentThread()
        start = datetime.now()
        IndexFiles(self.docDir, os.path.join(self.indexDir, self.indexFile), PorterStemmerAnalyzer(), self.yearList)  # PorterStemmer analyzer
        end = datetime.now()
        print end - start

if __name__ == '__main__':
    ## Experiment 6, index all MEDLINE docs
    time_span = sys.argv[1]
    print "Index MEDLINE %s..."%(time_span)
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    years = range(int(startyear),int(endyear)+1)

    data_dir = '/home/w2wei/data'
    data2_dir = '/home/w2wei/data2'
    medline_dir = os.path.join(data_dir, "medline_docs_by_year")
    tiab_base_dir = os.path.join(data2_dir, "tiab_by_year")
    mesh_base_dir = os.path.join(data2_dir, "mesh_by_year")
    if not os.path.exists(tiab_base_dir):
        os.makedirs(tiab_base_dir)
    if not os.path.exists(mesh_base_dir):
        os.makedirs(mesh_base_dir)

    # prepare TiAb text and MeSH for given years
    # for year in years:
    #     tiab_dir = os.path.join(tiab_base_dir,str(year))
    #     if not (os.path.exists(tiab_dir) and os.listdir(tiab_dir)):
    #         os.makedirs(tiab_dir)
    #         t0=time.time()
    #         extract_TiAb_from_MEDLINE(os.path.join(medline_dir,str(year)), tiab_dir)
    #         t1=time.time()
    #         print "Titles and abstracts for year %s ready: "%(year), t1-t0 
    #     mesh_dir = os.path.join(mesh_base_dir,str(year))
    #     if not (os.path.exists(mesh_dir) and os.listdir(mesh_dir)):
    #         os.makedirs(mesh_dir)
    #         t0=time.time()
    #         extract_MeSH_from_MEDLINE(os.path.join(medline_dir,str(year)), mesh_dir)
    #         t1=time.time()
    #         print "MeSH for year %s ready: "%(year), t1-t0 
                  

    # index MEDLINE of given years
    index_dir = os.path.join(data2_dir, "index")
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    if startyear!=endyear:
        index_file = "medline_%s_%s.index"%(startyear, endyear)
    else:
        index_file = "medline_%s.index"%(startyear)
    # if index_file not in os.listdir(index_dir):
    Index_new(tiab_base_dir, index_dir, index_file, years)
    # else:
        # print "%s already exists."%index_file    
    # ## Experiment 5, index general sub MEDLINE corpus
    # # print "Provide a start year and an end year, connected by -"
    # time_span = sys.argv[1]
    # print "Index MEDLINE %s..."%(time_span)
    # try:
    #     startyear, endyear = time_span.split("-")
    # except:
    #     startyear, endyear = time_span,time_span
    # years = range(int(startyear),int(endyear)+1)

    # data_dir = '/home/w2wei/data'
    # medline_dir = os.path.join(data_dir, "medline_docs_by_year")
    # tiab_base_dir = os.path.join(data_dir, "tiab")
    # mesh_base_dir = os.path.join(data_dir, "mesh")

    # # prepare TiAb text and MeSH for given years
    # if startyear!=endyear:
    #     tiab_dir = os.path.join(tiab_base_dir,"tiab_%s_%s"%(startyear,endyear))
    #     mesh_dir = os.path.join(mesh_base_dir,"mesh_%s_%s"%(startyear,endyear))
    # else:
    #     tiab_dir = os.path.join(tiab_base_dir,"tiab_%s"%(startyear))
    #     mesh_dir = os.path.join(mesh_base_dir,"mesh_%s"%(startyear))

    # if not (os.path.exists(tiab_dir) and os.listdir(tiab_dir)):
    #     os.makedirs(tiab_dir)
    #     t0=time.time()
    #     for year in years:
    #         extract_TiAb_from_MEDLINE(os.path.join(medline_dir,str(year)), tiab_dir)
    #     t1=time.time()
    #     print "Titles and abstracts for year %s to %s ready: "%(startyear,endyear), t1-t0 

    # if not (os.path.exists(mesh_dir) and os.listdir(mesh_dir)):
    #     os.makedirs(mesh_dir)
    #     t0=time.time()
    #     for year in years:
    #         extract_MeSH_from_MEDLINE(os.path.join(medline_dir,str(year)), mesh_dir)
    #     t1=time.time()
    #     print "MeSH for year %s to %s ready: "%(startyear,endyear), t1-t0         

    # # index MEDLINE of given years
    # index_dir = os.path.join(data_dir, "index")
    # if startyear!=endyear:
    #     index_file = "medline_%s_%s.index"%(startyear, endyear)
    # else:
    #     index_file = "medline_%s.index"%(startyear)
    # if index_file not in os.listdir(index_dir):
    #     Index(tiab_dir, index_dir, index_file)
    # else:
    #     print "%s already exists."%index_file

    # ## extract titles and abstracts from raw MEDLINE files
    # data_dir = '/home/w2wei/data'
    # latest_9M_medline_dir = os.path.join(data_dir, 'latest_9M_medline_docs')
    # latest_9M_tiab_dir = os.path.join(data_dir, 'latest_9M_tiab')
    # latest_9M_mesh_dir = os.path.join(data_dir, 'latest_9M_mesh')
    # t0=time.time()
    # extract_TiAb_from_MEDLINE(latest_9M_medline_dir, latest_9M_tiab_dir)
    # t1=time.time()
    # print "Prepare titles and abstracts ", t1-t0
    # extract_MeSH_from_MEDLINE(latest_9M_medline_dir, latest_9M_mesh_dir)
    # t2=time.time()
    # print "Prepare MeSH terms ",t2-t1

    # ## CALL Lucene to index the latest 9M MEDLINE
    # index_dir = os.path.join(data_dir, "index")
    # index_file = "latest_9M.index"
    # t0=datetime.now()
    # indexMED = Index(latest_9M_tiab_dir,index_dir,index_file) ## indexed documents in latest_3M_tiab_dir
    # t1=datetime.now()
    # print "Index time cost: ",t1-t0

    ## Experiment 2, index MEDLINE 1997
    # analysis_dir = os.path.join(data_dir, "latest_3M_analysis")
    # medline_1997_tiab_dir = os.path.join(analysis_dir, "tiab_1997")
    # medline_1997_index_file = "medline_1997.index"
    # index_dir = os.path.join(data_dir, "index")
    # t0=time.time()
    # index_medline_1997 = Index(medline_1997_tiab_dir, index_dir, medline_1997_index_file)
    # t1=time.time()
    # print t1-t0

    ## Experiment 3, index MEDLINE 1995_1997
    # analysis_dir = os.path.join(data_dir, "latest_3M_analysis")
    # medline_1995_1997_tiab_dir = os.path.join(analysis_dir, "tiab_1995_1997")
    # medline_1995_1997_index_file = "medline_1995_1997.index"
    # index_dir = os.path.join(data_dir, "index")
    # t0=time.time()
    # index_medline_1997 = Index(medline_1995_1997_tiab_dir, index_dir, medline_1995_1997_index_file)
    # t1=time.time()
    # print t1-t0

    ## Experiment 4, index MEDLINE 1993_1997
    # analysis_dir = os.path.join(data_dir, "latest_3M_analysis")
    # medline_1993_1997_tiab_dir = os.path.join(analysis_dir, "tiab_1993_1997")
    # medline_1993_1997_index_file = "medline_1993_1997.index"
    # index_dir = os.path.join(data_dir, "index")
    # t0=time.time()
    # index_medline_1997 = Index(medline_1993_1997_tiab_dir, index_dir, medline_1993_1997_index_file)
    # t1=time.time()
    # print t1-t0


