'''
    Predict the number of MeSH candidates

    Created on Sep 4, 2016
    Updated on Sep 4, 2016
    @author: Wei Wei

    The format of input data to BioASQ data as below
    {
     "username": "your_username", 
     "password": "your_password", 
     "documents": [
     {"pmid": 23476937, "labels": ["D000127", "D000128"]}
     ],
     "system": "your_system"
     }
'''

import numpy as np
import cPickle, os, string
import pandas as pd
from pprint import pprint
from collections import defaultdict
import subprocess

printable = set(string.printable)
replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))

def get_correct_num(pmid, std):
    data = zip(pmid, std)
    num_dict = defaultdict()
    for pmid, label in data:
        if not num_dict.get(pmid):
            num_dict[pmid]=label
        else:
            num_dict[pmid]+=label
    return num_dict

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
        rawMesh_meshAndEntry_dict = cPickle.load(file(os.path.join(outDir, "cap_rawMesh_cleanMeshAndEntry_dict.pkl"),"rb"))
    except:
        blocks = file(rawDataFile).read().split("\n\n")[:-1]
        rawMesh_meshAndEntry_dict = defaultdict() # {raw mesh: [clean mesh and clean entry terms]}
        for bk in blocks:
            lines = filter(None, bk.split("\n"))
            for entity in lines:
                if entity.startswith("MH = "):
                    mesh = entity.split("MH = ")[1] # raw mesh term
                    rawMesh_meshAndEntry_dict[mesh] = [mesh]
                if entity.startswith("ENTRY = "):
                    term = entity.split("ENTRY = ")[1].split("|")[0] # raw entry term
                    rawMesh_meshAndEntry_dict[mesh].append(term)
                if entity.startswith("PRINT ENTRY = "):
                    term = entity.split("PRINT ENTRY = ")[1].split("|")[0] # raw print entry term
                    rawMesh_meshAndEntry_dict[mesh].append(term)
        for rawMesh, meshAndEntry in rawMesh_meshAndEntry_dict.iteritems():
            rawMesh_meshAndEntry_dict[rawMesh] = mesh_parser(meshAndEntry)
        cPickle.dump(rawMesh_meshAndEntry_dict, file(os.path.join(outDir, "cap_rawMesh_cleanMeshAndEntry_dict.pkl"),"wb"))
    return rawMesh_meshAndEntry_dict

def loadMeshDict(rawDataFile, outDir):
    try:
        meshAndEntry_rawMesh_dict = cPickle.load(file(os.path.join(outDir, "meshAndEntry_rawMesh_dict.pkl")))
    except:
        rawMesh_meshAndEntry_dict = loadMeshEntryDict(rawDataFile, outDir)
        meshAndEntry_rawMesh_dict = {}
        for raw_mesh, clean_mesh_list in rawMesh_meshAndEntry_dict.iteritems():
            for mesh in clean_mesh_list:
                meshAndEntry_rawMesh_dict[mesh]=raw_mesh
        cPickle.dump(meshAndEntry_rawMesh_dict, file(os.path.join(outDir, "meshAndEntry_rawMesh_dict.pkl"),"w"))
    return meshAndEntry_rawMesh_dict

def loadMeshIdxDict(fname):
    basedir = os.path.dirname(fname)
    try:
        print 1.0/0
        mesh_idx_dict = cPickle.load(file(os.path.join(basedir, "mesh_index_dict.pkl")))
    except:
        mesh_idx_dict = defaultdict()
        fin = file(fname)
        for line in fin:
            mesh, idx = line.split("=")
            mesh_idx_dict[mesh]=idx.split("\r\n")[0]
        cPickle.dump(mesh_idx_dict, file(os.path.join(basedir, "mesh_index_dict.pkl"),"w"))
    return mesh_idx_dict

def select_cands(pmid, pred_prob, num_dict):
    data = zip(pmid, pred_prob)
    data_dict = defaultdict()
    cutoff_dict = {}

    for pmid, prob in data:
        if not data_dict.get(pmid):
            data_dict[pmid]=[prob]
        else:
            data_dict[pmid].append(prob)

    for pmid, prob_list in data_dict.iteritems():
        prob_list.sort(reverse=True)
        cutoff = num_dict.get(pmid)
        cutoff_dict[pmid]=prob_list[cutoff]

    all_preds = []
    for pmid, prob_list in data_dict.iteritems():
        pred = []
        cutoff = cutoff_dict.get(pmid)
        for prob in prob_list:
            if prob>cutoff:
                pred.append(1)
            else:
                pred.append(0)
        all_preds+=pred
    return np.array(all_preds)

def select_cands_by_percentage(pmid, pred_prob):
    print "select_cands_by_percentage"
    data = zip(pmid, pred_prob)
    data_dict = defaultdict()
    cutoff_dict = {}

    for pmid, prob in data:
        if not data_dict.get(pmid):
            data_dict[pmid]=[prob]
        else:
            data_dict[pmid].append(prob)

    for pmid, prob_list in data_dict.iteritems():
        prob_list.sort(reverse=True)
        cutoff = int(round(0.5*len(prob_list)))
        # cutoff=len(prob_list)-1
        print "cutoff ", cutoff 
        cutoff = prob_list[cutoff]
        print "cutoff ", cutoff 
        cutoff_dict[pmid]=cutoff#prob_list[cutoff]

    all_preds = []
    for pmid, prob_list in data_dict.iteritems():
        pred = []
        cutoff = cutoff_dict.get(pmid)
        for prob in prob_list:
            if prob>cutoff:
                pred.append(1)
            else:
                pred.append(0)
        all_preds+=pred
    return np.array(all_preds)

def main():
    exp = "Exp_38"
    pmid_file ='/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.qids.npy'%exp
    pmid_list = np.load(pmid_file)
    pred_prob_file = "/home/w2wei/projects/pointwiseLTR/src/deep-qa/exp.out/Exp_38;ndim=110;batch=5;max_norm=0;learning_rate=0.1;2016-09-01-12.39.02/test.epoch=05;batch=00012;dev_acc=87.24.predictions.npy"
    pred_prob = np.loadtxt(pred_prob_file)
    std_file = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.labels.npy"%exp
    std = np.load(std_file) ## gold standard labels

    # num_dict = get_correct_num(pmid_list, std)  
    # pred = select_cands(pmid_list, pred_prob, num_dict)
    pred = select_cands_by_percentage(pmid_list, pred_prob)

    # pred = []
    # for x in pred_prob:
    #     if x>=0.2:
    #         pred.append(1)
    #     else:
    #         pred.append(0)
    # pred = np.array(pred) ## predicted labels

    utils_dir = "/home/w2wei/projects/pointwiseLTR/data/utils/"
    raw_mesh_file = os.path.join(utils_dir,"d2016.bin")
    vocab_dir = os.path.join(utils_dir, "mesh_entry_vocab")
    mesh_idx_mapping_file = os.path.join(utils_dir, "MeSH_name_id_mapping_2016.txt")

    meshAndEntry_rawMesh_dict = loadMeshDict(raw_mesh_file, vocab_dir)    
    mesh_idx_dict = loadMeshIdxDict(mesh_idx_mapping_file)
    print "mesh_idx_dict: ", len(mesh_idx_dict)

    cand_mesh = cPickle.load(file("/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.cand_mesh.pkl"%exp))
    # test = meshAndEntry_rawMesh_dict.keys()
    # miss = list(set(cand_mesh)-set(test))
    cand_mesh = [meshAndEntry_rawMesh_dict.get(term) for term in cand_mesh]
    cand_mesh_idx = [mesh_idx_dict.get(term) for term in cand_mesh]

    std_mesh_dict = cPickle.load(file("/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.std_mesh.pkl"%exp))
    std_mesh_idx_dict = defaultdict()
    for pmid, mesh_list in std_mesh_dict.iteritems():
        raw_mesh_list = [meshAndEntry_rawMesh_dict.get(term) for term in mesh_list]
        mesh_idx_list = [mesh_idx_dict.get(term) for term in raw_mesh_list]
        std_mesh_idx_dict[pmid] = mesh_idx_list
    
    N = len(pred)
    print "pmid_list: ", len(pmid_list)
    print "pred_prob: ", len(pred_prob)
    print "N=",N
    result = pd.DataFrame(index=np.arange(N), columns=['pmid', 'pred', 'mesh'])
    result['pmid'] = pmid_list
    result['pred'] = pred
    result['mesh'] = cand_mesh_idx
    unique_pmids = np.unique(result['pmid'])
    print "unique pmid num: ", unique_pmids.shape

    outList = []
    idx = 0
    for pmid in unique_pmids:
        outDict = {}
        idx+=1
        if idx%1000==0:
            print idx
        local_data = result[result['pmid']==pmid]
        pred = local_data[local_data['pred']>=0.5]['mesh']
        outDict['pmid']=pmid
        outDict['labels']=filter(None, list(pred))
        outList.append(outDict)

    cand_mesh_idx_dict = defaultdict()
    for item in outList:
        cand_mesh_idx_dict[item['pmid']]=item['labels']


    ## output cand_mesh_idx_dict and std_mesh_idx_dict
    fsys = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/sys_mesh.txt"%exp
    fsys_mapped = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/sys_mesh_mapped.txt"%exp
    fstd = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/std_mesh.txt"%exp
    fstd_mapped = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/std_mesh_mapped.txt"%exp
    fout1 = file(fsys,"w")
    fout2 = file(fstd,"w")

    for pmid in unique_pmids:
        sys_mesh = cand_mesh_idx_dict[pmid]
        if not sys_mesh:
            continue
        sys_mesh = " ".join(sys_mesh)+'\n'
        fout1.write(sys_mesh)
        std_mesh = std_mesh_idx_dict[pmid]
        std_mesh = filter(None, std_mesh)
        std_mesh = " ".join(std_mesh)+'\n'
        fout2.write(std_mesh)
    print "fsys: ", fsys
    print "fsys_mapped: ", fsys_mapped
    print 'fstd: ', fstd
    print "fstd_mapped: ", fstd_mapped
    # subprocess.call("java -Xmx10G -cp ../bioasq_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar converters.MapMeshResults mesh/mapping.txt %s %s"%(fsys, fsys_mapped), shell=True)
    # subprocess.call("java -Xmx10G -cp ../bioasq_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar converters.MapMeshResults mesh/mapping.text %s %s"%(fstd, fstd_mapped), shell=True)
    # # subprocess.call("java -Xmx10G -cp ./bioasq_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.Evaluator %s %s"%(fstd_mapped, fsys_mapped), shell=True)

    # java -Xmx10G -cp ./bioasq_eval/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.Evaluator /home/w2wei/projects/pointwiseLTR/data/knn_sample/Exp_21/std_mesh_mapped.txt /home/w2wei/projects/pointwiseLTR/data/knn_sample/Exp_21/sys_mesh_mapped.txt
if __name__ == '__main__':
    main()


