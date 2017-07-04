'''This script evaluates the prediction from LeNet using the in-house evaluation tool. '''

import numpy as np
import pandas as pd
import cPickle, sys

def precision(data):
    '''micro P'''
    pred_true = data[data['pred']==1]
    tp = pred_true[pred_true['std']==1]
    prec = tp.shape[0]*1.0/pred_true.shape[0]
    return prec

def recall(data):
    '''micro R'''
    std_true = data[data['std']==1]
    pred_tp = std_true[std_true['pred']==1]
    recall = pred_tp.shape[0]*1.0/std_true.shape[0]
    return recall

def fscore(prec, reca):
    '''micro F1'''
    return 2*prec*reca/(prec+reca)

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

    print pred_prob
    print len(pred_prob)


    pred = []
    for x in pred_prob:
        if x>=0.5:
            pred.append(1)
        else:
            pred.append(0)
    pred = np.array(pred) ## predicted labels
    ## mesh vocab
    cand_mesh = cPickle.load(file("/home/w2wei/projects/pointwiseLTR/data/knn_sample/%s/test.cand_mesh.pkl"%exp))

    N = len(pred)
    result = pd.DataFrame(index=np.arange(N), columns=['pmid', 'pred', 'std', 'mesh'])
    result['pmid'] = pmid
    result['pred'] = pred
    result['std'] = std
    result['mesh'] = cand_mesh

    prec = precision(result)
    reca = recall(result)
    f1 = fscore(prec, reca)
    print "prec: ", prec
    print "reca: ", reca
    print "f1: ", f1
    # baseline 0.390 0.712 0.504 0.626
    # t=0.5 0.697, 0.456, 0.551 0.656
    # t=0.4 0.623, 0.539, 0.578 0.656
    # t=0.3 0.608, 0.564, 0.585 0.656
    # t=0.2 0.601, 0.578, 0.590 0.656

if __name__ == '__main__':
    main()

