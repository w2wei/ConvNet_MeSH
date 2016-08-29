'''
    Retrieve documents and rank them using BM25 algorithm and save pmids of 50 most similar articles. 
    
	Created on July 16, 2016
	Updated on August 7, 2016
	@author: Wei Wei
'''
import os, lucene, re, time, cPickle, sys
from knn_data import Data
from collections import defaultdict
from corpus_medline_index import PorterStemmerAnalyzer

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

# try:
#     lucene.initVM(vmargs=['-Djava.awt.headless=true']) # uncomment when run Retrieve separately
# except Exception as e:
#     print e

    
class Retrieve(object):
    '''This class retrieves documents from the indexed corpus'''
    def __init__(self, index_dir, index_file, query_dict):
        self.indexFile = os.path.join(index_dir, index_file)
        try:
            lucene.initVM(vmargs=['-Djava.awt.headless=true']) # uncomment when run Retrieve separately
        except Exception as e:
            print e
        directory = SimpleFSDirectory(File(self.indexFile))
        searcher = IndexSearcher(DirectoryReader.open(directory))
        searcher.setSimilarity(BM25Similarity(1.2, 0.75))  # set BM25 as the similarity metric, k=1.2, b=0.75"
        analyzer = PorterStemmerAnalyzer()
        self.result = defaultdict() # retrieved results, {pmid:[similar pmids]}
        self.submit(searcher, analyzer, query_dict)
        del searcher
        
    def run(self, searcher, analyzer, rawQuery):
        query = QueryParser(Version.LUCENE_CURRENT, "contents", analyzer).parse(QueryParser.escape(rawQuery))  # escape special characters 
        scoreDocs = searcher.search(query, 50).scoreDocs
        result = []
        for scoreDoc in scoreDocs:
            doc = searcher.doc(scoreDoc.doc)
            # result.append(doc.get("name").split(".txt")[0]) ## original
            result.append((doc.get("name").split(".txt")[0], scoreDoc.score)) ## updated
        return result   

    def submit(self, searcher, analyzer, query_dict):
        '''Query the indexed latest 3M MEDLINE'''
        for pmid, text in query_dict.iteritems():
            ret = self.run(searcher, analyzer, text)
            self.result[pmid] = ret

def load_lu_query_data(data_dir):
    '''Prepare queries for query from Lu data'''
    raw_data_dir = os.path.join(data_dir,"lu_data")
    clean_dir = os.path.join(data_dir, "lu_data", "clean")
    ## prepare queries
    L1000 = Data(raw_data_dir, clean_dir)
    L1000.large1000()
    NLM2007 = Data(raw_data_dir, clean_dir)
    NLM2007.nlm2007()
    SMALL200 = Data(raw_data_dir, clean_dir)
    SMALL200.small200()
    data_list = [L1000, NLM2007, SMALL200]

    L1000_pmids = L1000.query_pmids
    NLM2007_pmids = NLM2007.query_pmids
    SMALL200_pmids = SMALL200.query_pmids
    pmid_list = [L1000_pmids, NLM2007_pmids, SMALL200_pmids]

    L1000_query_dict = defaultdict()
    NLM2007_query_dict = defaultdict()
    SMALL200_query_dict = defaultdict()
    query_dict_list = [L1000_query_dict, NLM2007_query_dict, SMALL200_query_dict]

    for i in range(len(pmid_list)):
        pmids = pmid_list[i]
        for pmid in pmids:
            title, abstract, _ = data_list[i].query_tam[pmid]
            query_dict_list[i][pmid] = ". ".join(title + abstract)

    return query_dict_list

def submit_queries(query_dict, index_dir, index_file, knn_file):
    '''Query the indexed MEDLINE'''
    try:
        result = cPickle.load(file(knn_file))
    except:
        t0 = time.time()
        ret = Retrieve(index_dir, index_file, query_dict)
        result = ret.result
        print "result: ", len(result), type(result)
        t1=time.time()
        print "Time for %d queries: "%(len(query_dict)), t1-t0
        cPickle.dump(result, file(knn_file, "w"))

def submit_lu_queries(queries, index_dir, index_file, knn_file):
    '''Query the indexed MEDLINE'''
    try:
        # print 1.0/0
        ret_dict = cPickle.load(file(knn_file))
    except:
        name_list = ["L1000", "NLM2007", "SMALL200"]
        ret_dict= {}
        for i in xrange(len(queries)):
            name = name_list[i]
            print name
            query_dict = queries[i]
            t0 = time.time()
            ret = Retrieve(index_dir, index_file, query_dict)
            print "result: ", len(ret.result), type(ret.result)
            ret_dict[name] = ret.result
            t1=time.time()
            print "Time for %d queries: "%(len(query_dict)), t1-t0
        # cPickle.dump(ret_dict, file(knn_file, "w"))
    return ret_dict

if __name__ == '__main__':
    data_dir = '/home/w2wei/data'
    data2_dir = '/home/w2wei/data2'
    index_dir = os.path.join(data2_dir, "index")
    time_span = sys.argv[1]
    try:
        startyear, endyear = time_span.split("-")
    except:
        startyear, endyear = time_span,time_span
    if startyear!=endyear:
        time_span = "%s_%s"%(startyear, endyear)
    else:
        time_span = startyear
    index_file = "medline_%s.index"%(time_span)
    print "Query %s"%index_file

    knn_result_base_dir = os.path.join(data2_dir, "knn")
    if not os.path.exists(knn_result_base_dir):
        os.makedirs(knn_result_base_dir)


    ## prepare queries
    L1000_query_dict, NLM2007_query_dict, SMALL200_query_dict = load_lu_query_data(data_dir)
    ## Exp 2: submit L1000 as query
    query_name = "L1000"
    print "Query data: ", query_name
    query_dict = L1000_query_dict
    knn_result_dir = os.path.join(knn_result_base_dir, "%s_on_MEDLINE_%s"%(query_name, time_span))
    if not os.path.exists(knn_result_dir):
        os.makedirs(knn_result_dir)
    print "knn_result_dir: ", knn_result_dir
    knn_result_file = os.path.join(knn_result_dir, "%s_knn_on_MEDLINE_%s.pkl"%(query_name, time_span))
    submit_queries(query_dict, index_dir, index_file, knn_result_file)

    ## Exp 1: submit NLM2007 as query
    # query_name = "NLM2007"
    # print "Query data: ", query_name
    # knn_result_dir = os.path.join(knn_result_base_dir, "%s_on_MEDLINE_%s"%(query_name, time_span))
    # print "knn_result_dir: ", knn_result_dir
    # if not os.path.exists(knn_result_dir):
    #     os.makedirs(knn_result_dir)
    # print "Result in %s"%knn_result_dir

    # query_dict = NLM2007_query_dict
    # knn_result_file = os.path.join(knn_result_dir, "%s_knn_on_MEDLINE_%s.pkl"%(query_name, time_span))
    # submit_queries(query_dict, index_dir, index_file, knn_result_file)

    ## original experiment    
    # data_dir = '/home/w2wei/data'
    # index_dir = os.path.join(data_dir, "index")
    # index_file = "latest_3M.index"
    # knn_dir = os.path.join(data_dir, "knn_from_clean")
    # if not os.path.exists(knn_dir):
    #     os.makedirs(knn_dir)
    # knn_3M_file = os.path.join(knn_dir, "lu_data_knn_from_3M.pkl") ## KNN results output file
    # ## prepare queries
    # L1000_query_dict, NLM2007_query_dict, SMALL200_query_dict = load_lu_query_data(data_dir)
    # ## submit queries from Lu data
    # ret_dict = submit_lu_queries([L1000_query_dict, NLM2007_query_dict, SMALL200_query_dict], index_dir, index_file, knn_3M_file)
    # for datasetname, value in ret_dict.iteritems():
    #     print datasetname, len(value)
