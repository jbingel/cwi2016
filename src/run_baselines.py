import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn import dummy, tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from cwi_util import readSentences
from feats_and_classify import WordInContext

import numpy as np

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defaultdata = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Baselines")
    parser.add_argument('--train', help="parsed-and-label input format", default=defaultdata)
    args = parser.parse_args()

    labels = []
    featuredicts = []

    print("Collecting features...")
    count=0
    for s in readSentences(args.train):
       print("\r"+str(count), end="")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordInContext(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"])
                featuredicts.append(w.baselinefeatures())
                labels.append(w.label)
    print()
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    classifiers = [LogisticRegression(penalty='l1'),LogisticRegression(penalty='l2'),SGDClassifier(),tree.DecisionTreeClassifier(),dummy.DummyClassifier(strategy="most_frequent") ]


    scores = defaultdict(list)

    for classifier in classifiers:
        for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=1):
            TrainX_i = features[TrainIndices]
            Trainy_i = labels[TrainIndices]

            TestX_i = features[TestIndices]
            Testy_i =  labels[TestIndices]

            classifier.fit(TrainX_i,Trainy_i)
            ypred_i = classifier.predict(TestX_i)

            scores["Accuracy"].append(accuracy_score(ypred_i,Testy_i))
            scores["F1"].append(f1_score(ypred_i,Testy_i))
            scores["Precision"].append(precision_score(ypred_i,Testy_i))
            scores["Recall"].append(recall_score(ypred_i,Testy_i))
        print("--", str(classifier))
        for key in sorted(scores.keys()):
            currentmetric = np.array(scores[key])
            print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))

    sys.exit(0)



if __name__ == "__main__":
    main()
