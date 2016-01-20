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

def getBestThreshold(features, labels_pooled,labels_current):
    print("length of pooled and current",len(labels_pooled),len(labels_current))
    maxent = LogisticRegression(penalty='l1')
    scores = {"F1":[], "Recall":[], "Accuracy":[], "Precision":[]}
    thresholds=[]

    print('Finding best thresholds...')
    fold=1
#    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=2, shuffle=False, random_state=None):
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=10, shuffle=False, random_state=None):
#    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        print('\r'+str(fold), end="")
        fold+=1
        TrainX_i = features[TrainIndices]
        Trainy_i = labels_pooled[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels_current[TestIndices]

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

def predictAcrossThresholds(features, labels_pooled,labels_current, maxent, thresholds, average=True, median=False):
    if average:
        print('Using average of best threshold...')
        t=thresholds.mean()
        score_dict_ave=predictWithThreshold(features, labels_pooled,labels_current, maxent, t)
    if median:
        print('Using median of best threshold...')
        t=np.median(thresholds)
        score_dict_med=predictWithThreshold(features, labels_pooled,labels_current, maxent, t)
    return score_dict_ave, score_dict_med

def predictWithThreshold(features, labels_pooled,labels_current, maxent, threshold):
    out_dict = {}
    scores = defaultdict(list)
    fold=1
#    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=2, shuffle=False, random_state=None):
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels_pooled, n_folds=10, shuffle=False, random_state=None):
#    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        print('\r'+str(fold), end="")
        fold+=1
        TrainX_i = features[TrainIndices]
        Trainy_i = labels_pooled[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels_current[TestIndices]

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
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016--optimisation of threshhold")
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--annotator',type=str,default="03")

    args = parser.parse_args()
    current_single_ann = scriptdir+"/../data/cwi_training/cwi_training_"+args.annotator+".lbl.conll"
    testfile = scriptdir+"/../data/cwi_testing/cwi_testing.txt.conll"
    X_train, y_train, v_train = feats_and_classify.collect_features(current_single_ann)
    X_test, y_test, v_test = feats_and_classify.collect_features(current_single_ann)
    maxent = LogisticRegression(penalty='l1')
    maxent.fit(X_train,y_train)
    y_pred_proba = maxent.predict_proba(X_test)
    ypred_i=["1" if pair[1]>=args.threshold else "0" for pair in y_pred_proba]
    fout = open(args.annotator+".pred",mode="w")
    print(" ".join(ypred_i),file=fout)
    fout.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
