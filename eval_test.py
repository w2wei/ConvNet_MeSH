import numpy as np 
import pandas as pd
from collections import defaultdict

def get_precision(data):
    pmidList = list(set(data['pmid'])) # unique pmids
    cands = []
    for pmid in pmidList:
        sub = data[data['pmid']==pmid]
        sub = sub.sort_values(['pred'], ascending=False)
        sub = sub[:2]
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

        sub_p = sub[:2]
        sub_p.ix[:,'pred']=1
        sub_n = sub[2:]
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

def get_map(data):
    pmidList = list(set(data['pmid'])) # unique pmids
    cands = []
    ap_list = []
    for pmid in pmidList:
        sub = data[data['pmid']==pmid]
        sub = sub.sort_values(['pred'], ascending=False)
        sub_p = sub[:2]
        sub_p.ix[:,'pred']=1
        sub_n = sub[2:]
        sub_n.ix[:,'pred']=0
        sub = pd.concat([sub_p, sub_n])
        ap = get_ap(sub)
        ap_list.append(ap)
    MAP = sum(ap_list)*1.0/len(ap_list)
    print "AP list"
    print ap_list
    return MAP
  

def get_ap(data):
    # data = np.array(data[['pred','std']])
    N = data.shape[0]
    # print "N=", N
    AN = data[data['std']==1].shape[0]
    # print "AN=", AN
    ap = 0

    for i in range(1,N+1):
        cutoff = data.iloc[i-1:i,1]
        # print "cutoff"
        if int(cutoff)==0:
            continue        
        # print "========"
        # print "i=",i
        # print "data"
        # print data
        # print
        sub = data[0:i]
        # print "sub"
        # print sub
        # print        
        pred_true = sub[sub['pred']==1]
        # print "pred true "
        # print pred_true
        # print
        correct_pred = pred_true[pred_true['std']==1]
        # print "correct pred"
        # print correct_pred
        # print
        prec_i = correct_pred.shape[0]*1.0/i
        # print "prec at i: ",prec_i
        # print

        ap+=prec_i
    ap = ap*1.0/AN
    # print "AP: ", ap
    # print data
    # print 
    return ap
    
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

if __name__=="__main__":
    ## sample data
    # data = {"pmid":['1','1','1','1','2','2','2','2','3','3','3','3','3'],
    #         "pred":[0.1,0.5,0.7,0.9,0.4,0.2,0.8,0.6,0.9, 0.9,0.9,0.9,0.1],
    #         "std":[ 0,  1,  0,  1,  1,  0,  1,  1, 0,  0,  1,  0,  1]
    #         }
    data = {"pmid":['1','1','1','1','2','2','2','2','3','3','3','3','3'],
            "pred":[0.1,0.5,0.7,0.9,0.4,0.2,0.8,0.6,0.9,0.4,0.5,0.7,0.2],
            "std":[ 0,  1,  0,  1,  1,  0,  1,  1,  1,  0,  1,  0,  1  ]
            }            
    df = pd.DataFrame(data)
    print df

    prec = get_precision(df)
    print "prec: ", prec

    # recall = get_recall(df)
    # print "recall: ", recall

    # MAP = get_map(df)
    # print "map: ", MAP

    print map_score(data['pmid'],data['std'],data['pred'])
