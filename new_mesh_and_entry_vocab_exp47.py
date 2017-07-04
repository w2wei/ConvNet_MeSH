'''
    Build s vocabulary of MeSH terms and associated entry terms.

    Directories and paths are for new servers.

    Created on Sep 11, 2016
    Updated on Sep 11, 2016
    @author Wei Wei
'''

import os, re, string, pickle, time
from collections import defaultdict, Counter
printable = set(string.printable)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

def mesh_parser(termList):
    newTermList = []
    for term in termList:
        text = filter(lambda x: x in printable, term) 
        text = text.translate(replace_punctuation)
        text = " ".join(text.split()).lower()
        text = text.replace(" ", "_")
        newTermList.append(text)
    return newTermList

def loadMeshEntryDict(rawDataFile, outDir):
    '''Load a {mesh_term: entry_term} dictionary'''
    try:
        # print 1.0/0
        rawMesh_meshAndEntry_dict = pickle.load(file(os.path.join(outDir, "rawMesh_cleanMeshAndEntry_dict.pkl"),"rb"))
    except:
        blocks = file(rawDataFile).read().split("\n\n")[:-1]
        rawMesh_meshAndEntry_dict = defaultdict() # {raw mesh: [clean mesh and clean entry terms]}
        for bk in blocks:
            lines = filter(None, bk.split("\n"))
            for entity in lines:
                if entity.startswith("MH = "):
                    mesh = entity.split("MH = ")[1].lower() # raw mesh term
                    rawMesh_meshAndEntry_dict[mesh] = [mesh]
                if entity.startswith("ENTRY = "):
                    term = entity.split("ENTRY = ")[1].split("|")[0] # raw entry term
                    rawMesh_meshAndEntry_dict[mesh].append(term)
                if entity.startswith("PRINT ENTRY = "):
                    term = entity.split("PRINT ENTRY = ")[1].split("|")[0] # raw print entry term
                    rawMesh_meshAndEntry_dict[mesh].append(term)
        for rawMesh, meshAndEntry in rawMesh_meshAndEntry_dict.iteritems():
            rawMesh_meshAndEntry_dict[rawMesh] = mesh_parser(meshAndEntry)
        pickle.dump(rawMesh_meshAndEntry_dict, file(os.path.join(outDir, "rawMesh_cleanMeshAndEntry_dict.pkl"),"wb"))
    return rawMesh_meshAndEntry_dict

def parse(rawDataFile, outDir):
    vocab = [] # [human, confounding varialbe]
    phrase_vocab = [] # [confounding variable]
    phrase_token_vocab = [] # [confounding_variable]
    phrase_idx = defaultdict() ## an index of phrases
    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    
    fin = file(rawDataFile).readlines()
    for line in fin:
        term = ''
        if line.startswith("MH = "):
            term = re.findall("MH = (.*?)[\n]",line)[0]
        if line.startswith("ENTRY = "):
            term = re.findall("ENTRY = (.*?)[\n|\|]", line)[0]
        if line.startswith("PRINT ENTRY ="):
            term = re.findall("PRINT ENTRY = (.*?)[\n|\|]", line)[0]
        term = term.translate(replace_punctuation)
        term = filter(None, term.split(" "))
        vocab.append(" ".join(term).lower())
    vocab = filter(None, vocab)
    vocab = list(set(vocab))
    vocab.sort()
    
    for term in vocab:
        if " " in term: # phrase mesh
            phrase_vocab.append(term) ## build the phrase mesh vocab
            phrase_token_vocab.append(term.replace(" ", "_"))
            tokenList = term.split(" ")
            if phrase_idx.get(tokenList[0]):
                phrase_idx[tokenList[0]]+=[term]
            else:
                phrase_idx[tokenList[0]]=[term]

    for token, termList in phrase_idx.iteritems():
        phrase_idx[token] = max(map(len,[term.split(" ") for term in termList]))

    phrase_list = zip(phrase_vocab, phrase_token_vocab) # mesh_phrase_token_vocab
    phrase_dict = defaultdict(None,phrase_list)

    pickle.dump([vocab, phrase_idx, phrase_dict], file(os.path.join(outDir, "all_mesh_and_entry.pkl"),"wb"))
    return [vocab, phrase_idx, phrase_dict]

def loadAll():
    '''Load all '''
    ## For idash-data
    # utils_dir = "/home/w2wei/data/nlm_data"
    # raw_mesh_file = os.path.join(utils_dir,"d2016.bin")
    # vocab_dir = os.path.join(utils_dir, "mesh_entry_vocab")    
    ## For idash-cloud
    # utils_dir = "/home/w2wei/projects/pointwiseLTR/data/utils/"
    utils_dir = "/home/w2wei/data/utils"
    raw_mesh_file = os.path.join(utils_dir,"d2016.bin")
    vocab_dir = os.path.join(utils_dir, "mesh_entry_vocab")
    # ##  For idash-meta-dev
    # raw_mesh_file = "/home/w2wei/Research/mesh/data/NLM/d2016.bin"
    # vocab_dir = "/home/w2wei/Research/mesh/data/deep_pmcoa/mesh_entry_vocab"
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)    
    try:
        vocab, phrase_idx, phrase_dict = pickle.load(file(os.path.join(vocab_dir, "all_mesh_and_entry.pkl"),"rb"))
    except Exception as e:
        vocab, phrase_idx, phrase_dict = parse(raw_mesh_file, vocab_dir)

    rawMesh_meshAndEntry_dict = loadMeshEntryDict(raw_mesh_file, vocab_dir)

    return [vocab, phrase_idx, phrase_dict, rawMesh_meshAndEntry_dict]

if __name__ == "__main__":
    ## For new VMs
    utils_dir = "/home/w2wei/data/utils/"
    raw_mesh_file = os.path.join(utils_dir,"d2016.bin")
    vocab_dir = os.path.join(utils_dir, "mesh_entry_vocab")  
    ##  For idash-meta-dev  
    # raw_mesh_file = "/home/w2wei/Research/mesh/data/NLM/d2016.bin"
    # vocab_dir = "/home/w2wei/Research/mesh/data/deep_pmcoa/mesh_entry_vocab"
    t0=time.time()
    vocab, phrase_idx, phrase_dict, rawMesh_meshAndEntry_dict = loadAll()
    t1=time.time()
    print "time ", t1-t0
    # for k, v in rawMesh_meshAndEntry_dict.iteritems():
    #     print k
    #     print v
    #     raw_input("...")

    # print "mesh and entry term vocab size: ", len(vocab)
    # print "mesh phrase index size: ",len(phrase_idx)
    # print "phrase_dict size: ", len(phrase_dict)
    # print "rawMesh_meshAndEntry_dict size: ", len(rawMesh_meshAndEntry_dict)
    # print phrase_token_vocab[:10]