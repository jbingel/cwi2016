import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *
import porter, pickle
import numpy as np
#from resources import *
import argparse, feats_and_classify
    
def optimize_threshold(maxent,TestX_i,Testy_i):
    best_t = 0
    best_f1 = -1
    best_ypred_i=None
    t_results = {}
    for thelp in range(1000):
        t=thelp/1000.0
        ypred_i, t_results[t] = pred_for_threshold(maxent, TestX_i, Testy_i, t)
        f1 = t_results[t][0]
        if f1 > best_f1:
            best_t = t
            best_f1 = f1
            best_ypred_i=ypred_i
    return best_t, best_ypred_i, t_results[best_t]

def pred_for_threshold(maxent,TestX_i,Testy_i, t):
    #ypred_i = maxent.predict(TestX_i)
    ypred_probs=maxent.predict_proba(TestX_i)
    
    ypred_i=[1 if pair[1]>=t else 0 for pair in ypred_probs]
    
    acc = accuracy_score(ypred_i, Testy_i)
    pre = precision_score(ypred_i, Testy_i)
    rec = recall_score(ypred_i, Testy_i)

    # shared task uses f1 of *accuracy* and recall!
    f1 = 2 * acc * rec / (acc + rec)
    return ypred_i, (f1, rec, acc, pre)

def getBestThreshold(features, labels_pooled,y_current, regularization='l2'):
    print("length of pooled and current",len(labels_pooled),len(y_current))
    maxent = LogisticRegression(penalty=regularization)
    scores = {"F1":[], "Recall":[], "Accuracy":[], "Precision":[]}
    thresholds=[]

    print('Finding best thresholds...')
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=10, shuffle=False, random_state=None):
        print('\r'+str(fold), end="")
        fold+=1
        TrainX_i = features[TrainIndices]
        Trainy_i = labels_pooled[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  y_current[TestIndices]

        maxent.fit(TrainX_i,Trainy_i)
        #get prediction
        thresh_i, ypred_i, score=optimize_threshold(maxent,TestX_i,Testy_i)
        thresholds.append(thresh_i)

        scores["F1"].append(score[0])
        scores["Recall"].append(score[1])
        scores["Accuracy"].append(score[2])
        scores["Precision"].append(score[3])
    
    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("\n--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")
    
    return maxent, np.array(thresholds)

def predictAcrossThresholds(features, labels_pooled,y_current, maxent, thresholds, average=True, median=False):
    if average:
        print('Using average of best threshold...')
        t=thresholds.mean()
        score_dict_ave=predictWithThreshold(features, labels_pooled,y_current, maxent, t)
    if median:
        print('Using median of best threshold...')
        t=np.median(thresholds)
        score_dict_med=predictWithThreshold(features, labels_pooled,y_current, maxent, t)
    return score_dict_ave, score_dict_med

def predictWithThreshold(features, labels_pooled,y_current, maxent, threshold):
    out_dict = {}
    scores = defaultdict(list)
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=10, shuffle=False, random_state=None):
        print('\r'+str(fold), end="")
        fold+=1
        TrainX_i = features[TrainIndices]
        Trainy_i = labels_pooled[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  y_current[TestIndices]

        ypred_i, score=pred_for_threshold(maxent,TestX_i,Testy_i, threshold)

        scores["F1"].append(score[0])
        scores["Recall"].append(score[1])
        scores["Accuracy"].append(score[2])
        scores["Precision"].append(score[3])

    
    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("\n--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        out_dict[key] = (currentmetric.mean(),currentmetric.std())
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")
    return out_dict


def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    default_pool = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016--optimisation of threshhold")
    parser.add_argument('--pooled_annotators', help="parsed-and-label input format", default=default_pool)
    parser.add_argument('--threshold_matrix_file', help="location/name of the threshold matrix", default='annotator_threshold_matrix')
    parser.add_argument('--regularization', help="regularizer, may be l1 or l2", default='l2')
    args = parser.parse_args()

    # f1_matrix holds for every training annotator: the list of tuples of 
    # avg/med f1_row based on avg/med threshold
    f1_matrix = []
    # holds for every training annotator: the list of tuples of avg/med threshold
    t_matrix = []
    current_label_list = []
    
    f1_final = [] # holds 4-tuples of avgs over (f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med) f.e. tr 
    t_final  = [] # holds 4-tuples of (t_avg_avg, t_avg_med, t_med_avg, t_med_med) f.e. tr

    # train for every annotator...
    #for trainidx in "05 08 15".split(" "):
    #for trainidx in "05".split(" "):
    for trainidx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
        current_single_ann = scriptdir+"/../data/cwi_training/cwi_training_"+trainidx+".lbl.conll"
        X_current, y_current_tr, v_current = feats_and_classify.collect_features(current_single_ann)
        print("Training on Annotator "+trainidx)
        f1_row = [] # holds 4-tuples of (f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med) f.e. tr/te
        t_row  = [] # holds 2-tuples of (t_avg, t_med) f.e. tr/te
        f1_matrix.append(f1_row)
        t_matrix.append(t_row)
        
        # optimize t for every annotator (except training annotator), yields avg/med t 
        for idx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
        #for idx in "02 03 04".split(" "):
            if idx == trainidx:
                continue
            print("  Testing on annotator "+idx)
            current_single_ann = scriptdir+"/../data/cwi_training/cwi_training_"+idx+".lbl.conll"

            _, y_current_te, _ = feats_and_classify.collect_features(current_single_ann)
            current_label_list.append(y_current_te)

            maxent, thresholds=getBestThreshold(X_current, y_current_tr, y_current_te, regularization=args.regularization)
            t_avg = np.average(thresholds)
            t_med = np.median(thresholds)
            t_row.append((t_avg, t_med))
        
        # calculate avg of avg t's, avg of med t's, ... for the current training annotator
        t_avg_avg = np.average([t[0] for t in t_row]) 
        t_avg_med = np.average([t[1] for t in t_row]) 
        t_med_avg =  np.median([t[0] for t in t_row]) 
        t_med_med =  np.median([t[1] for t in t_row]) 
        t_final.append((t_avg_avg, t_avg_med, t_med_avg, t_med_med))
 
        maxent = LogisticRegression(penalty=args.regularization)
        maxent.fit(X_current, y_current_tr) 
        for idx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
            if idx == trainidx:
                continue

            f1_avg_avg = predictWithThreshold(X_current, y_current_tr, y_current_te, maxent, t_avg_avg)['F1'][0]
            f1_avg_med = predictWithThreshold(X_current, y_current_tr, y_current_te, maxent, t_avg_med)['F1'][0]
            f1_med_avg = predictWithThreshold(X_current, y_current_tr, y_current_te, maxent, t_med_avg)['F1'][0]
            f1_med_med = predictWithThreshold(X_current, y_current_tr, y_current_te, maxent, t_med_med)['F1'][0]

            f1_row.append((f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med))

        f1_final.append(tuple(map(np.average, zip(*f1_row))))

    print(f1_final)
    # get the index (NB: array index!) of the max avg/med F1 (i.e. computed on avg/med threshold)
    best_trainidx_avg_avg = np.argmax([f1[0] for f1 in f1_final])
    best_trainidx_avg_med = np.argmax([f1[1] for f1 in f1_final])
    best_trainidx_med_avg = np.argmax([f1[2] for f1 in f1_final])
    best_trainidx_med_med = np.argmax([f1[3] for f1 in f1_final])

    print(t_final)

    sys.exit(0)


if __name__ == "__main__":
    main()
