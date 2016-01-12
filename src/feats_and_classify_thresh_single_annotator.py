import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *
import porter
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
    maxent = LogisticRegression(penalty='l1')
    scores = {"F1":[], "Recall":[], "Accuracy":[], "Precision":[]}
    thresholds=[]

    print('Finding best thresholds...')
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=False, random_state=None):
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

def predictAcrossThresholds(features, labels, maxent, thresholds, average=True, median=False):
    if average:
        print('Using average of best threshold...')
        t=thresholds.mean()
        predictWithThreshold(features, labels, maxent, t)
    if median:
        print('Using median of best threshold...')
        t=np.median(thresholds)
        predictWithThreshold(features, labels, maxent, t)

def predictWithThreshold(features, labels, maxent, threshold):
    scores = defaultdict(list)
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(labels, n_folds=10, shuffle=False, random_state=None):
#    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        print('\r'+str(fold), end="")
        fold+=1
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels[TestIndices]

        ypred_i, score=pred_for_threshold(maxent,TestX_i,Testy_i, threshold)

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


def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    default_pool = scriptdir+"/../data/cwi_training/cwi_training_txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016--optimisation of threshhold")
    parser.add_argument('--pooled_annotators', help="parsed-and-label input format", default=default_pool)
    args = parser.parse_args()

    features, labels_pooled, vec = feats_and_classify.collect_features(args.pooled_annotators)

    for idx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
        current_single_ann = scriptdir+"/../data/cwi_training/cwi_training_"+idx+".lbl.conll"

        _, labels_current, _ = feats_and_classify.collect_features(current_single_ann)
    
        maxent, thresholds=getBestThreshold(features, labels_pooled,labels_current)
        print(thresholds)
        #predictAcrossThresholds(features, labels, maxent, thresholds, average=True, median=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
