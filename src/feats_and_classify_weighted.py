import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation, tree, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *
import porter
import numpy as np
import math
from feats_and_classify import WordInContext


def get_sample_weights(npa,mode):
    if mode == "uniform":
        return np.array([1]*len(npa))
    elif mode == "inverse_class_relevance":
        outvec = []
        C = Counter(npa)
        for v in npa:
            outvec.append((sum(C.values()) - C[v]) / sum(C.values()))
        return np.array(outvec)
    elif mode == "log_and_mode":
        C = Counter([x for x in npa if x > 0])
        for k in C.keys():
            C[k] = math.log(C[k]+1)
        mode_value=C.most_common(n=1)[0][1]
        C[0]=mode_value
        outvec = [C[x] for x in npa]
        return np.array(outvec)
    elif mode == "log_and_max":
        C = Counter([x for x in npa if x > 0])
        for k in C.keys():
            C[k] = math.log(C[k]+1)
        C[0]=min(C.values())
        outvec = [C[x] for x in npa]
        return np.array(outvec)
    elif mode == "tf_idf":
        C = Counter([x for x in npa if x > 0])
        idf = Counter()
        sample_weight = Counter()
        for k in C:
            idf[k] = math.log((max(npa))/k)
            sample_weight[k] = C[k] * idf[k] / sum(C.values())

        mode_value=sample_weight.most_common(n=1)[0][1]
        sample_weight[0] = mode_value
        outvec = [sample_weight[x] for x in npa]
        return np.array(outvec)
    elif mode == "linear":
        outvec = []
        C = Counter([x for x in npa if x > 0])
        mode_key = mode_value=C.most_common(n=1)[0][0]
        for x in npa:
            if x == 0:
                outvec.append(mode_key)
            else:
                outvec.append(x)
        return np.array(outvec)








def main():
    #global brownclusters, cluster_heights, ave_brown_depth, ave_brown_height, max_brown_depth, embeddings
    #brownclusters, cluster_heights, ave_brown_depth, ave_brown_height, max_brown_depth=read_brown_clusters('/coastal/brown_clusters/rcv1.64M-c1000-p1.paths', 1000)
    #embeddings=read_embeddings('/coastal/mono_embeddings/glove.6B.300d.txt.gz')

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defaultdata = scriptdir+"/../data/cwi_training/cwi_training_allannotations.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=defaultdata)
    parser.add_argument('--instance_weighting', choices=["uniform","linear","inverse_class_relevance","log_and_mode","tf_idf","log_and_max"], default="uniform")
    args = parser.parse_args()

    labels = []
    featuredicts = []
    
    #print("Collecting features...")
    count=0
    positive_votes = []
    for s in readSentences(args.train):
       #print("\r"+str(count), end="")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordInContext(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],positive_votes=l,heads=s["head"],deprels=s["deprel"])
                featuredicts.append(w.featurize())
                labels.append(w.label)
                positive_votes.append(w.positive_votes)
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    positive_votes = np.array(positive_votes)

    learners = [tree.DecisionTreeClassifier(), svm.NuSVC(nu=0.2)]
    for learner in learners:
        scores = defaultdict(list)
        for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):

            TrainX_i = features[TrainIndices]
            Trainy_i = labels[TrainIndices]
            sampleweights_i = get_sample_weights(positive_votes[TrainIndices], args.instance_weighting)
            #print(sampleweights_i)
            TestX_i = features[TestIndices]
            Testy_i =  labels[TestIndices]

            learner.fit(TrainX_i,Trainy_i,sample_weight=sampleweights_i)
            ypred_i = learner.predict(TestX_i)

            scores["Accuracy"].append(accuracy_score(ypred_i,Testy_i))
            scores["F1"].append(f1_score(ypred_i,Testy_i))
            scores["Precision"].append(precision_score(ypred_i,Testy_i))
            scores["Recall"].append(recall_score(ypred_i,Testy_i))

        print("--")
        print(learner)
        for key in sorted(scores.keys()):
            currentmetric = np.array(scores[key])
            print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
        print("--")





if __name__ == "__main__":
    main()
