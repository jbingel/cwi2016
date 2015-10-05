import argparse
from sklearn.linear_model.logistic import LogisticRegression
from collections import Counter
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn


class WordInContext:
    def __init__(self,sentence,word,index,label):
        self.sentence = sentence.strip().split()
        self.word = word
        self.index = int(index)
        self.label = int(label)

    def a_simple_feats(self): #
        D = {}
        D["a_form"] = self.word
        D["a_formlength"] = len(self.word)
        D["a_relativeposition"] =  self.index / len(self.sentence)
        return D


    def b_wordnet_feats(self): #
        D = {}
        D["b_nsynsets"] = len(wn.synsets(self.word))
        return D

    def featurize(self):
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        return D

def prettyprintweights(linearmodel,vectorizer):
   for name, value in zip(vectorizer.feature_names_, linearmodel.coef_[0]):
       print("\t".join([name,str(value)]))


def main():
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="first parse (prob. Malt)", default="../data/cwi_training/cwi_training.txt")
    args = parser.parse_args()


    parsedsentences = []
    labels = []
    featuredicts = []

    for line in open(args.train).readlines():
        sentence, word,idx, label =  line.strip().split("\t")
        w = WordInContext(sentence, word,idx, label)
        featuredicts.append(w.featurize())
        labels.append(w.label)


    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()

    maxent = LogisticRegression()
    maxent.fit(features,labels)
    prettyprintweights(maxent,vec)
    coeffs = list(maxent.coef_[0])
    lowest = min(coeffs)
    highest = max(coeffs)
    print("--")
    print("lowest coeff:",lowest, vec.feature_names_[coeffs.index(lowest)])
    #print(lowest, vec.feature_names_[coeffs.index(lowest)])
    print("highest coeff",highest, vec.feature_names_[coeffs.index(highest)])




    #
    # features = []
    # annotations = []
    #




if __name__ == "__main__":
    main()
