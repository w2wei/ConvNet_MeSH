from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize,TreebankWordTokenizer, wordpunct_tokenize
from nltk.corpus import stopwords
import string, re, pickle
import numpy as np



class Analyzer(object):
    def run(self,text):
        sent_list = self.sent_tokenizer(text)
        stemmed_text = []
        if not sent_list:
            return
        else:
            porter_stemmer = PorterStemmer()
            for words in sent_list:
                words = self.digitFilter(self.letterFilter(self.stopFilter(self.puncFilter(self.tokenizer(words)))))
                words = [porter_stemmer.stem(word) for word in words]
                stemmed_text.append(" ".join(words))
        return stemmed_text

    def sent_tokenizer(self,text):
        return sent_tokenize(text.strip().lower(), "english") # tokenize an article text

    def tokenizer(self, sent):
        sent = re.sub("[-*&]"," ",sent) # remove hyphen
        return TreebankWordTokenizer().tokenize(sent) # tokenize the sentence

    def puncFilter(self, words):
        return [word.strip(string.punctuation) for word in words]

    def stopFilter(self, words):
        return [word for word in words if not word in stopwords.words("english")]

    def letterFilter(self, words):
        return [word for word in words if len(word)>1]

    def digitFilter(self, words):
        return [word for word in words if re.search('[a-zA-Z]',word)]

    def porterStemmer(self, words):
        return [porter_stemmer.stem(word) for word in words] # apply Porter stemmer

def parseMHlist(text):
    mhList = text.split("!")
    mhList = [mh.strip("*") for mh in mhList]
    return mhList

def parseMH_pssd(text):
    '''Tokenize, regularize and normalize extracted MeSH terms'''
    text = re.sub("[-*&]"," ",text)
    text = text.strip("*")
    analyzer = Analyzer()
    psdText = analyzer.run(text)[0]
    return psdText

def parseMHlist_pssd(text):
    '''Tokenize, regularize and normalize extracted a MeSH term list'''
    text = re.sub("[-*&]"," ",text)
    mhList = text.split("!")
    mhList = [mh.strip("*") for mh in mhList]
    analyzer = Analyzer()
    pssdMhList = [analyzer.run(term)[0] for term in mhList]
    return pssdMhList

def getNbrMhVocab(nbr_pmids, nbr_text):
    '''Collect MeSH terms from the neighbor articles of the given PMIDs'''
    nbr_mh_list = []
    for pmid in nbr_pmids:
        try:
            curr_nbr_vocab = parseMHlist_pssd(nbr_text[pmid][2]) # the MeSH vocabulary of the current nbr pmid
        except:
            curr_nbr_vocab = []
        nbr_mh_list.append(curr_nbr_vocab)
        # nbr_mh_vocab = list(set([item for sublist in nbr_mh_list for item in sublist]))
    nbr_mh_vocab = list(set([item for sublist in nbr_mh_list for item in sublist]))        
    return [nbr_mh_vocab, nbr_mh_list]

def loadMeshDics():
    '''Load the MeSH-Tree dict and Tree-MeSH dict '''
    treeRawMhDict = {}
    rawMeshTrDict = {}
    # treePsdMhDict = {}
    # psdMeshTrDict = {}
    try:
        treeRawMhDict = pickle.load(file(tree_raw_mesh_dict_file,"rb"))
        rawMeshTrDict = pickle.load(file(raw_mesh_tree_dict_file,"rb"))
        # treePsdMhDict = pickle.load(file(tree_psd_mesh_dict_file,"rb"))
        # psdMeshTrDict = pickle.load(file(psd_mesh_tree_dict_file,"rb"))
    except:
        fmhtree = file(raw_mesh2016_file).read()
        content = filter(None, fmhtree.split("\n\n"))
        for rawmesh in content:
            sub = rawmesh.split("\n")
            curr_mesh = ''
            curr_psd_mesh = ''
            for line in sub:
                if line.startswith("MH = "):
                    curr_mesh = line.split("MH = ")[1].lower()
                    curr_psd_mesh = parseMH_pssd(line.split("MH = ")[1].lower())
                    try:
                        rawMeshTrDict[curr_mesh]=[]
                        psdMeshTrDict[curr_psd_mesh]=[]
                    except Exception as e:
                        print "MeSH-Tree dict build error: ", e
                if line.startswith("MN = "):
                    tree = line.split("MN = ")[1]
                    rawMeshTrDict[curr_mesh].append(tree)
                    psdMeshTrDict[curr_psd_mesh].append(tree)
                    try:
                        treeRawMhDict[tree] = curr_mesh
                        treePsdMhDict[tree] = curr_psd_mesh
                    except Exception as e:
                        print "Tree-MeSH dict build error: ", e
        pickle.dump(treeRawMhDict, file(tree_raw_mesh_dict_file,"wb"))
        pickle.dump(rawMeshTrDict, file(raw_mesh_tree_dict_file,"wb"))
        # pickle.dump(treePsdMhDict, file(tree_psd_mesh_dict_file,"wb"))
        # pickle.dump(psdMeshTrDict, file(psd_mesh_tree_dict_file,"wb"))
    return [treeRawMhDict,rawMeshTrDict]#,treePsdMhDict,psdMeshTrDict]

def loadMeshEntryDics():
    '''Load the MeSH-Entry term dict and Entry_term - MeSH dict. All terms are pre-processed.'''
    try:
        rawMeshEntryDict = pickle.load(file(raw_mesh_entry_dict_file, "rb"))
        rawEntryMeshDict = pickle.load(file(raw_entry_mesh_dict_file, "rb"))
    except:
        rawEntryMeshDict = {} # {entry:mesh}
        rawMeshEntryDict = {} # {mesh:entry}
        fmhtree = file(raw_mesh2016_file).read()
        content = filter(None, fmhtree.split("\n\n"))
        for rawmesh in content:
            sub = rawmesh.split("\n")
            currRawEntry = ''
            currPssdEntry = ''
            for line in sub:
                if line.startswith("MH = "):
                    currRawMesh = line.split("MH = ")[1].lower()
                    try:
                        rawMeshEntryDict[currRawMesh]=[]
                    except Exception as e:
                        print "MeSH-EntryTerm dict build error: ", e
                if line.startswith("ENTRY ="):
                    entry = line.split("ENTRY = ")[1].split("|")[0].lower()
                    rawMeshEntryDict[currRawMesh].append(entry)
                    try:
                        rawEntryMeshDict[entry].append(currRawMesh)
                    except:
                        rawEntryMeshDict[entry]=[currRawMesh]
        pickle.dump(rawMeshEntryDict, file(raw_mesh_entry_dict_file, "wb"))
        pickle.dump(rawEntryMeshDict, file(raw_entry_mesh_dict_file, "wb"))
    ## pre-process two dictionaries
    try:
        pssdMeshEntryDict = pickle.load(file(pssd_mesh_entry_dict_file,"rb"))
        pssdEntryMeshDict = pickle.load(file(pssd_entry_mesh_dict_file,"rb"))
    except:
        pssdEntryMeshDict = {}
        pssdMeshEntryDict = {}
        for mesh, entryList in rawMeshEntryDict.iteritems():
            pssd_mesh = parseMH_pssd(mesh)
            pssd_entries = []
            for entry in entryList:
                try:
                    pssd_entry = parseMH_pssd(entry)
                    pssd_entries.append(pssd_entry)
                except Exception as e:
                    pass
            pssd_entries = list(set(pssd_entries))
            pssdMeshEntryDict[pssd_mesh]=pssd_entries
        
        for entry, meshList in rawEntryMeshDict.iteritems():
            try:
                pssd_entry = parseMH_pssd(entry)
            except:
                pass
            if pssd_entry not in pssdEntryMeshDict.keys():
                pssdEntryMeshDict[pssd_entry] = []
            pssdEntryMeshDict[pssd_entry] += [parseMH_pssd(mesh) for mesh in meshList]
        for entry, meshList in pssdEntryMeshDict.iteritems():
            pssdEntryMeshDict[pssd_entry] = list(set(meshList))
        pickle.dump(pssdMeshEntryDict, file(pssd_mesh_entry_dict_file, "wb"))
        pickle.dump(pssdEntryMeshDict, file(pssd_entry_mesh_dict_file, "wb"))
    return [rawMeshEntryDict, rawEntryMeshDict, pssdMeshEntryDict, pssdEntryMeshDict]

def loadCheckTags():
    '''Load raw check tags'''
    tagList = file(nlm_checktag_file).read().split("\n")
    analyzer = Analyzer()
    pssd_checktags = [analyzer.run(tag) for tag in tagList]
    pssd_checktags = [item for sublist in pssd_checktags for item in sublist]
    return pssd_checktags

def build_NonCT_TrainSet(data, feature, trainDataFile):
    '''Generate a training set for non-checktag MeSH terms'''
    trainData = {} ## the output training data
    pssd_checktags = loadCheckTags()
    maxFeatures = [1, feature.nbr_pssd_qMH_matrix.max(), feature.nbr_pssd_sim_matrix.max(), feature.unigram_overlap_matrix.max(), \
                      feature.bigram_overlap_matrix.max(), feature.okapi_matrix.max(), feature.translation_matrix.max(), \
                      feature.synonym_entry_matrix.max(), feature.synonym_unigram_matrix.max()] # feature.model_bias_matrix.max()
    qPmidList = data.query_pmids ## all query PMIDs in data
    for q_pmid in qPmidList:
        trainData[q_pmid] = []
        q_pmid_index = data.query_pmids.index(q_pmid) # index of current PMID
        q_mhs = parseMHlist_pssd(data.query_tam[q_pmid][2]) ## pre-processed gold std MeSH terms

        q_mhs = list(set(q_mhs)-set(pssd_checktags)) # exclude check tags
        
        nbr_pmids = [p[0] for p in feature.nbr20_dict[q_pmid]] # 20 nearest neighbor PMIDs
        nbr_mh_list = [] # a list of lists. [[mh1, mh2, ...]], the order of mesh lists is the order of the vocabulary in each pmid
        for pmid in nbr_pmids:
            try:
                curr_nbr_mesh = parseMHlist_pssd(data.nbr_tam[pmid][2])
            except:
                curr_nbr_mesh = []
            nbr_mh_list.append(curr_nbr_mesh)
        nbr_mhs = list(set([item for sublist in nbr_mh_list for item in sublist]))

        nbr_mhs = list(set(nbr_mhs) - set(pssd_checktags)) ## exclude check tags from MeSH candidates
        
        nbr_mhs.sort()

        trainDataMatrix = np.zeros((len(nbr_mhs), len(maxFeatures)))
        trainLabelVec = np.zeros((len(nbr_mhs),2)) # 2 columns of labels, one for original labels, one for exponentialized.                  
        for mh in nbr_mhs:
            mh_index = feature.p_mesh_vocab.index(mh)
            local_mh_index = nbr_mhs.index(mh)         
            f1 = feature.nbr_pssd_qMH_matrix[q_pmid_index,mh_index]/maxFeatures[1] ## nbr feature 1
            f2 = feature.nbr_pssd_sim_matrix[q_pmid_index,mh_index]/maxFeatures[2] ## nbr feature 2
            f3 = feature.unigram_overlap_matrix[q_pmid_index,mh_index]/maxFeatures[3] ## unigram
            f4 = feature.bigram_overlap_matrix[q_pmid_index,mh_index]/maxFeatures[4] ##  bigram
            f5 = feature.okapi_matrix[q_pmid_index,mh_index]/maxFeatures[5] ## okapi
            f6 = feature.translation_matrix[q_pmid_index, mh_index]/maxFeatures[6] ## translation
            f7 = feature.synonym_entry_matrix[q_pmid_index, mh_index]/maxFeatures[7] ## synonym entry
            f8 = feature.synonym_unigram_matrix[q_pmid_index, mh_index]/maxFeatures[8] ## synonym unigram
            trainDataMatrix[local_mh_index, 0] = 1
            trainDataMatrix[local_mh_index, 1] = f1 ## nbr feature 1
            trainDataMatrix[local_mh_index, 2] = f2 ## nbr feature 2
            trainDataMatrix[local_mh_index, 3] = f3 ## unigram
            trainDataMatrix[local_mh_index, 4] = f4 ## bigram
            trainDataMatrix[local_mh_index, 5] = f5 ## okapi
            trainDataMatrix[local_mh_index, 6] = f6 ## translation
            trainDataMatrix[local_mh_index, 7] = f7 ## synonym 1
            trainDataMatrix[local_mh_index, 8] = f8 ## synonym 2
            trainLabelVec[local_mh_index,0] = (mh in q_mhs)
        trainLabelVec[:,1] = np.exp(trainLabelVec[:,0])
        gs_sum = trainLabelVec[:,1].sum()
        trainLabelVec[:,1] = np.divide(trainLabelVec[:,1], gs_sum)
        trainData[q_pmid] = [trainDataMatrix,trainLabelVec,q_mhs,nbr_mhs]
    # pickle.dump([trainData, maxFeatures], file(trainDataFile,"wb"))
    return [trainData, maxFeatures]
