'''This script evaluates the prediction from LeNet using the in-house evaluation tool. '''

import numpy as np
import pandas as pd
import cPickle, sys
from collections import defaultdict

def micro(data):
    mesh_voc = list(np.unique(data['mesh']))
    micro_data = []
    idx = 0
    for mesh in mesh_voc:
        idx+=1
        if idx%1000==0:
            print idx
        local_data = data[data['mesh']==mesh]
        # print "Current MeSH: ", mesh
        tp = local_data[(local_data['pred']==1) & (local_data['std']==1)].shape[0]
        fp = local_data[(local_data['pred']==1) & (local_data['std']==0)].shape[0]
        fn = local_data[(local_data['pred']==0) & (local_data['std']==1)].shape[0]
        tn = local_data[(local_data['pred']==0) & (local_data['std']==0)].shape[0]
        micro_data.append([tp, fp, fn, tn])
    outData = np.array(micro_data)
    outDF = pd.DataFrame(index=np.arange(len(mesh_voc)), columns=['mesh', 'tp', 'fp','fn','tn'])
    outDF['mesh']=mesh_voc
    outDF[['tp','fp','fn','tn']]=micro_data
    # print "performance matrix"
    # print outDF
    # print "="*10
    tp_sum = outDF['tp'].sum()
    print "tp sum: ", tp_sum
    fp_sum = outDF['fp'].sum()
    print "fp sum: ", fp_sum
    fn_sum = outDF['fn'].sum()
    print "fn sum: ", fn_sum
    micro_prec = tp_sum*1.0/(tp_sum+fp_sum)
    print "micro_prec: ", micro_prec
    micro_reca = tp_sum*1.0/(tp_sum+fn_sum)
    print "micro_recall: ", micro_reca
    micro_f1 = 2*micro_prec*micro_reca*1.0/(micro_prec+micro_reca)
    print "micro_f1: ", micro_f1

def get_precision(data):
    pmidList = list(set(data['pmid'])) # unique pmids
    cands = []
    for pmid in pmidList:
        sub = data[data['pmid']==pmid]
        sub = sub.sort_values(['pred'], ascending=False)
        sub = sub[:25]
        sub['pred']=1
        cands.append(sub)
    cands = pd.concat(cands)
    tp = cands[cands['std']==1]
    prec = tp.shape[0]*1.0/cands.shape[0]
    return prec

def get_recall(data):
    pmidList = list(set(data['pmid'])) # unique pmids
    cands = []
    for pmid in pmidList:
        sub = data[data['pmid']==pmid]
        sub = sub.sort_values(['pred'], ascending=False)

        sub_p = sub[:25]
        sub_p.ix[:,'pred']=1
        sub_n = sub[25:]
        sub_n.ix[:,'pred']=0
        sub = pd.concat([sub_p, sub_n])
        cands.append(sub)
    cands = pd.concat(cands)
    std_true = cands[cands['std']==1]
    pred_tp = std_true[std_true['pred']==1]
    recall = pred_tp.shape[0]*1.0/std_true.shape[0]
    return recall

def fscore(prec, reca):
    return 2*prec*reca/(prec+reca)

def map_score(qids, labels, preds):
    qid2cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        qid2cand[qid].append((pred, label))

    average_precs = []
    for qid, candidates in qid2cand.iteritems():
        average_prec = 0
        running_correct_count = 0
        for i, (score, label) in enumerate(sorted(candidates, reverse=True), 1):
            if label > 0:
                running_correct_count += 1
                average_prec += float(running_correct_count) / i
        average_precs.append(average_prec / (running_correct_count + 1e-6))
    map_score = sum(average_precs) / len(average_precs)
    return map_score

def main():
    exp = "Exp_35"
    pred_file = "Exp_35;ndim=110;batch=5;max_norm=0;learning_rate=0.1;2016-08-29-11.19.50/test.epoch=11;batch=00012;dev_acc=87.58.predictions.npy"

    # Load data
    pmid_file ='/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.qids.npy'%exp
    pmid = np.load(pmid_file)

    pred_prob_file = "/home/w2wei/projects/pointwiseLTR/src/deep-qa/exp.out/%s"%(pred_file)
    pred_prob = np.loadtxt(pred_prob_file)
    std_file = "/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.labels.npy"%exp
    std = np.load(std_file) ## gold standard labels

    ## mesh vocab
    cand_mesh = cPickle.load(file("/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.cand_mesh.pkl"%exp))

    N = len(pred_prob)
    result = pd.DataFrame(index=np.arange(N), columns=['pmid', 'pred', 'std', 'mesh'])
    result['pmid'] = pmid
    result['pred'] = pred_prob
    result['std'] = std
    result['mesh'] = cand_mesh

    ## select top 25 candidates for every PMID
    # result = select_candidates(result)
    prec = get_precision(result)
    print "prec: ", prec

    ## recall
    recall = get_recall(result)
    print "recall: ", recall

    f1 = fscore(prec, recall)
    print "f1: ", f1

    MAP = map_score(result['pmid'],result['std'],result['pred'])
    print "map: ", MAP
    # baseline 0.390 0.712 0.504 0.626
    # t=0.5 0.697, 0.456, 0.551 0.656
    # t=0.4 0.623, 0.539, 0.578 0.656
    # t=0.3 0.608, 0.564, 0.585 0.656
    # t=0.2 0.601, 0.578, 0.590 0.656

if __name__ == '__main__':
    main()

