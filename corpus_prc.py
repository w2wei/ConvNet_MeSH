'''
	Implement Jimmy Lin and John Wilbur's PRC algorithm to find K nearest neighbors from the latest 1M MEDLINE records.

	Before running this script, install PyLucene (https://lucene.apache.org/pylucene/install.html).
	1. Download pylucence installer.
	2. Before install JCC, install python-dev. Command: apt-get install python-dev. This will set up c++ Python.h etc. \
	   make sure JCC is correctly installed. Check JAVA_HOME, java version
	3. Install ANT. Check ANT_HOME, etc



	Created on July 15, 2016
	Updated on July 15, 2016
	@author: Wei Wei
'''
import os, lucene, threading, re
import multiprocessing as mp
from Bio import Entrez
from datetime import datetime
from IndexFiles import IndexFiles
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

Entrez.email = "granitedewint@gmail.com"

class PorterStemmerAnalyzer(PythonAnalyzer):
    def createComponents(self, fieldName, reader):
        source = StandardTokenizer(Version.LUCENE_CURRENT, reader)
        filter = StandardFilter(Version.LUCENE_CURRENT, source)
        filter = LowerCaseFilter(Version.LUCENE_CURRENT, filter)  # normalize token text to lower case
        filter = PorterStemFilter(filter)  # transform the token stream as per the Porter stemming algorithm
        filter = StopFilter(Version.LUCENE_CURRENT, filter,
                            StopAnalyzer.ENGLISH_STOP_WORDS_SET)
        return self.TokenStreamComponents(source, filter)
    
class Consumer(mp.Process):
    def __init__(self, task_queue):  # result_queue
        mp.Process.__init__(self)
        self.task_queue = task_queue
        
    def run(self):
        '''Split texts into sentences for word2vec'''
#         proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print "%s: Exiting" % mp.current_process()
                self.task_queue.task_done()
                break
#             print "%s: %s"%(proc_name,next_task)
            next_task.__call__()
#             answer = next_task.__call__()
            self.task_queue.task_done()
#             self.result_queue.put(answer)
        return
    
class Task(object):
    def __init__(self, inFile, outFile):
        self.inputFile = inFile
        self.outputFile = outFile
    
    def __call__(self):
        sentences = []
        text = file(self.inputFile).read()
        sent_tokenize_list = sent_tokenize(text.strip().lower(), "english")  # a sentence list from doc 
        if sent_tokenize_list:  # if sent_tokenize_list is not empty
            porter_stemmer = PorterStemmer()
            for sent in sent_tokenize_list:                
                words = TreebankWordTokenizer().tokenize(sent) # tokenize the sentence
                words = [word.strip(".") for word in words]
                words = [word for word in words if not word in stopwords.words("english")]
                words = [word for word in words if len(word)>1] # remove single letters and non alphabetic characters
                words = [word for word in words if re.search('[a-zA-Z]',word)]
                words = [porter_stemmer.stem(word) for word in words]                
                sentences.append(words)
        self.__save__(sentences)
    
    def __save__(self, sentences):
        fout = file(self.outputFile, "w")
        texts = [" ".join(sent) for sent in sentences]
        fout.write("\n".join(texts))

    def __str__(self):
        return "%s " % (self.inputFile)

def collectMEDLINE(base_dir, gensim_dir):
    print "Splitting texts..."
    tasks = mp.JoinableQueue()
#     results = mp.Queue()
    
    num_consumers = mp.cpu_count() * 2
    print "creating %d consumers " % num_consumers
    consumers = [Consumer(tasks) for i in xrange(num_consumers)]
    
    for w in consumers:
        w.start()
    
    for doc in os.listdir(gensim_dir):
        inFile = os.path.join(gensim_dir, doc)
        outFile = os.path.join(base_dir, "gensim", "allMH_sentences", doc)
        tasks.put(Task(inFile, outFile))
        
    for i in xrange(num_consumers):
        tasks.put(None)
        
    tasks.join()

class Corpus(object):
    '''This class collects MEDLINE and store results in a database.'''
    def __init__(self, baseDir, pmidFile, medlineDir, databaseDir):
        self.baseDir = baseDir
        self.pmidFile = pmidFile
        self.medDir = medlineDir
        self.dbDir = databaseDir
        self.pmid = []
    
    def run(self):
        self.prepRelPMID()
        if not os.listdir(self.medDir):
            self.prepMED()  # collect MEDLINE if not existing
#         if not os.listdir(self.dbDir):
        self.prepDatabase()
    
#     def collectAllMED(self):
#         self.pmid = file(self.pmidFile).read().split("\n")
#         if not os.listdir(self.medDir):
#             self.prepMED()
    
    def prepRelPMID(self):
        '''extract 4584 relevant PMID from the raw PMID document'''
        rawPMID = file(self.pmidFile)
        for line in rawPMID:
            cat, label, pmid, rel = line.split("\t")
            if rel[:-1] != "0":  # if rel is possible relevant or definitely relevant
                self.pmid.append(pmid)
        self.pmid = list(set(self.pmid))  # 11706583 no long exists, 4490 unique PMID, 4583 in all.
    
    def prepMED(self):
        '''collect MEDLINE'''
        retstart = 0
        step = 5000
        i = 0
        ite = len(self.pmid) / step * 1.0 + 1
        while i < ite:  #
            print "Iteration ",i
            query = ",".join(self.pmid[retstart:retstart + step])
            handle = Entrez.efetch(db='pubmed', id=query, rettype="medline", retmodel="text")
            record = handle.read()     
            handle.close()
            # output
            outFile = os.path.join(self.medDir, "%d_medline_iter_%d.txt" % (len(self.pmid),i))
            fout = file(outFile, "w")
            fout.write(record)
            # update index
            retstart += step
            i += 1

    def prepDatabase(self):
        '''extract information from medline, and prepare text for indexing'''
#         fout = file(os.path.join(self.baseDir, "query.txt"), "w") # to reset query.txt
#         fout.write("") # to reset query.txt
        for doc in os.listdir(self.medDir):
            tiab = self.extractTiAb(os.path.join(self.medDir, doc))
            mesh = self.extractMH(os.path.join(self.medDir, doc))
            self.saveTiAbMH(tiab, mesh)  # write files to disk
#             self.prepQueries(tiab) # this may not 
    
    def prepQueries(self, TiAbDict):
        '''extract abstracts from medline, and save them'''
        print "prepare queries..."
        fout = file(os.path.join(self.baseDir, "query.txt"), "a")
        for pmid in TiAbDict.keys():
            absStr = TiAbDict[pmid][1]
            rec = "\n".join([pmid, absStr]) + "\n\n"
            fout.write(rec)
            
    def saveTiAbMH(self, TiAbDict, MeSHDict):
        for pmid in MeSHDict.keys():
            print pmid
            meshStr = ". ".join(MeSHDict[pmid]) + ". "
            titleStr = (TiAbDict[pmid][0].strip(".") + ". ") * 2
            absStr = TiAbDict[pmid][1].strip(".") + ". "
            text = meshStr + titleStr + absStr
            fout = file(os.path.join(self.dbDir, pmid), "w")
            fout.write(text)

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
                mhRaw = filter(None, mhRaw.split("/"))
                if len(mhRaw)>1: # qualifier exists
                    mhRaw = mhRaw[:1]+mhRaw
                mhList += mhRaw
                mhList = [item.strip("*") for item in mhList]
        
            if (not firstLine) and line == "\n":
                textDict[pmid] = mhList
                newEntry = False
            
            if lineCount == totalLine:  # check if this is the end of the file
                textDict[pmid] = mhList
                newEntry = False     
        return textDict
      
class Index(object):
    '''This class indexes prepared MEDLINE documents'''
    def __init__(self, doc_dir, index_dir, index_file):
        self.docDir = doc_dir
        self.indexDir = index_dir
        self.indexFile = index_file
        env = lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        if self.indexFile not in os.listdir(self.indexDir):
            threading.Thread(target=self.worker, args=(env,)).start()
        
    def worker(self, env):
        env.attachCurrentThread()
        start = datetime.now()
        if 'Standard' in self.indexFile:
            print "Use the StandardAnalyzer for indexing"
            IndexFiles(self.docDir, os.path.join(self.indexDir, self.indexFile), StandardAnalyzer(Version.LUCENE_CURRENT))  # StandardAnalyzer
        if 'Porter' in self.indexFile:
            print "Use the PorterStemmer analyzer for indexing"
            IndexFiles(self.docDir, os.path.join(self.indexDir, self.indexFile), PorterStemmerAnalyzer())  # PorterStemmer analyzer
        end = datetime.now()
        print end - start

class Retrieve(object):
    '''This class retrieves documents from the indexed corpus'''
    def __init__(self, index_dir, index_file, rawQuery):
        self.indexFile = os.path.join(index_dir, index_file)

#         lucene.initVM(vmargs=['-Djava.awt.headless=true']) # uncomment when run Retrieve separately
        directory = SimpleFSDirectory(File(self.indexFile))
        searcher = IndexSearcher(DirectoryReader.open(directory))
        searcher.setSimilarity(BM25Similarity(1.2, 0.75))  # set BM25 as the similarity metric, k=1.2, b=0.75
        if 'Standard' in self.indexFile:
            print "Use the StandardAnalyzer"
            analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)  # build a standard analyzer with default stop words
        if 'Porter' in self.indexFile:
            print "Use the PorterStemmer analyzer"
            analyzer = PorterStemmerAnalyzer()
        self.run(searcher, analyzer, rawQuery)
        del searcher
        
    def run(self, searcher, analyzer, rawQuery):
        query = QueryParser(Version.LUCENE_CURRENT, "contents", analyzer).parse(QueryParser.escape(rawQuery))  # escape special characters 
        scoreDocs = searcher.search(query, 50).scoreDocs
        print "%s total matching documents." % len(scoreDocs)

        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            print 'path:', doc.get("path"), 'name:', doc.get("name")

if __name__ == '__main__':
    base_dir = "/home/w2wei/Research/mesh/data/TREC/2005/4584rel"
    rawPMID_dir = "/home/w2wei/Research/mesh/data/TREC/2005/genomics.qrels.large.txt"
    medline_dir = os.path.join(base_dir, "medline")
    database_dir = os.path.join(base_dir, "database")
    index_dir = os.path.join(base_dir, "index")
#     analyzerType = "Standard"
    analyzerType = "PorterStemmer"
    index_file = "4584_MEDLINE_%s.index" %(analyzerType)
    
    # PREPARE DOCUMENTS
    corpus = Corpus(base_dir, rawPMID_dir, medline_dir, database_dir)
    corpus.run()
    print "4584 MEDLINE records are prepared"
    
    # CALL Lucene to index 4584 MEDLINE documents
#     t0=datetime.now()
#     indexMED = Index(database_dir,index_dir,index_file)
#     t1=datetime.now()
#     print "Index time cost: ",t1-t0

    # Retrieve documents: test if the index works. For retrieval function, use Search_TREC2005Genomics4584
#     query = file(os.path.join(database_dir,"7977635")).read()
#     ret = Retrieve(index_dir, index_file, query)
