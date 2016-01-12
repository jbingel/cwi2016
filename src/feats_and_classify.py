import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
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
import argparse

class WordInContext:
    def __init__(self,sentence,index,word,lemma,pos,namedentity,positive_votes,heads,deprels):
        self.sentence = sentence #sentence is a list of forms
        self.word = word
        self.index = int(index)
        self.positive_votes = int(positive_votes)
        self.label = int(self.positive_votes > 0)
        self.lemma = lemma
        self.pos = pos
        self.a_namedentity = namedentity
        self._heads = [int(h) for h in heads] #"protected" property
        self._deprels = deprels             #"protected" property
        self.deptree = self._create_deptree() #remember to use self.index+1 to access self.deptree

    def _create_deptree(self):
        deptree = nx.DiGraph()
        for idx_from_zero,head in enumerate(self._heads):
            #deptree.add_node(idx_from_zero+1) # the Id of each node is its Conll index (i.e. from one)
            deptree.add_edge(head, idx_from_zero+1, deprel=self._deprels[idx_from_zero]) #edge from head to dependent with one edge labeling called deprel
        return deptree

    def a_simple_feats(self): #
        D = {}
        #D["a_form"] = self.word
        #D["a_lemma"] = self.lemma
        D["a_pos"] = self.pos
        D["a_namedentity"] = self.a_namedentity
        D["a_formlength"] = len(self.word)
        return D

    def a_simple_feats_lexicalized(self): #
        D = {}
        D["a_form"] = self.word
        D["a_lemma"] = self.lemma
        D["a_pos"] = self.pos
        D["a_namedentity"] = self.a_namedentity
        D["a_formlength"] = len(self.word)
        return D


    def b_wordnet_feats(self): #
        D = {}
        D["b_nsynsets"] = len(wn.synsets(self.word))
        return D

    def c_positional_feats(self):
        D = {}
        D["c_relativeposition"] =  int(self.index / len(self.sentence))
        before, after = commas_before_after(self.sentence, self.index)
        D["c_preceding_commas"] = before
        D["c_following_commas"] = after
        before, after = verbs_before_after(self.sentence, self.index)
        D["c_preceding_verbs"] = before
        D["c_following_verbs"] = after
        return D
        
    def d_frequency_feats(self):
        D = {}
        wProb = prob(self.word, corpus="wp")
        wProbSimple = prob(self.word, corpus="swp")
        D["d_freq_in_swp"] = wProbSimple
        D["d_freq_in_wp"] = wProb
        D["d_freq_ratio_swp/wp"] = wProbSimple / wProb
        # TODO D["d_freqrank_distance_swp/wp"] = rank(self.word, corpus="swp") - rank(self.word, corpus="wp")  
        # TODO D["d_distributional_distance_swp/wp"] = dist(dist_vector(self.word, "swp"), dist_vector(self.word, "wp"))  # get distributional vector from background corpora, use some dist measure
        return D

    def e_morphological_feats(self):
        D = {}
        etymology = retrieve_etymology(self.lemma)
        D["e_latin_root"] = has_ancestor_in_lang("lat", etymology)  # check wiktionary
        D["e_length_dist_lemma_form"] = len(self.word) - len(self.lemma)
        stem, steps = porter.stem(self.word)
        D["e_length_dist_stem_form"] = len(self.word) - len(stem)
        D["e_inflectional_morphemes_count"] = steps 
        return D

    def f_prob_in_context_feats(self):
        D = {}
        # TODO D["f_P(w|w-1)"]    = seq_prob(self.word, [self.sentence[self.index-1]]) # prob() uses freq()
        # TODO D["f_P(w|w-2w-1)"] = seq_prob(self.word, [self.sentence[self.index-2:self.index]]) # prob() uses freq()
        # TODO D["f_P(w|w+1)"]    = seq_prob(self.word, [self.sentence[self.index+1]]) # prob() uses freq()
        # TODO D["f_P(w|w+1w+2)"] = seq_prob(self.word, [self.sentence[self.index+1:self.index+3]]) # prob() uses freq()
        return D
    
    def g_char_complexity_feats(self):
        D = {}
        unigramProb = prob(self.word, level="chars", order=1)
        unigramProbSimple = prob(self.word, level="chars", corpus="swp", order=1)
        bigramProb = prob(self.word, level="chars", order=2)
        bigramProbSimple = prob(self.word, level="chars", corpus="swp", order=2)
        D["g_char_unigram_prob"] = unigramProb
        D["g_char_unigram_prob_ratio"] = unigramProbSimple / unigramProb
        D["g_char_bigram_prob"] = bigramProb
        D["g_char_bigram_prob_ratio"] = bigramProbSimple / bigramProb
        D["g_vowels_ratio"] = float(count_vowels(self.word)) / len(self.word)
        return D 
    
    def h_brownpath_feats(self):
        D={}
        #brown cluster path feature
        #global brownclusters
        if self.word in brownclusters:
            D["h_cluster"] = brownclusters[self.word]
        else:
            D["h_nocluster"]=1
        return D
        
    def i_browncluster_feats(self):
        D={}
        #brown cluster path feature
        #global brownclusters, ave_brown_height, ave_brown_depth
        if self.word in brownclusters:
            bc = brownclusters[self.word]
            for i in range(1,len(bc)):
                D["i_cluster_"+bc[0:i] ]=1
        
            #brown cluster height=general/depth=fringiness
            D["i_cluster_height"]=len(bc)
            D["i_cluster_depth"]=cluster_heights[bc]
        else:
            #taking average
            #D["i_cluster_height"]=ave_brown_height
            #D["i_cluster_depth"]=ave_brown_depth
            #taking extremes
            D["i_nocluster"]=0
            D["i_cluster_height"]=0
            D["i_cluster_depth"]=max_brown_depth
        return D
        
    def j_embedding_feats(self):
        D={}
        #word embedding
        if self.word in embeddings.keys():
            emb=embeddings[self.word]
            for d in range(len(emb)):
                D["j_embed_"+str(d)]=float(emb[d])
            else:
                D["j_noembed"]=1

        #TODO: (1) fringiness of embedding 
        return D

    def k_dependency_feats(self):
        wordindex = self.index + 1
        headindex = dep_head_of(self.deptree,wordindex)
        D = {}
        D["k_dist_to_root"] = len(dep_pathtoroot(self.deptree,wordindex))
        D["k_deprel"] = self.deptree[headindex][wordindex]["deprel"]
        D["k_headdist"] = abs(headindex - wordindex) # maybe do 0 for root?
        D["k_head_degree"] = nx.degree(self.deptree,headindex)
        D["k_child_degree"] = nx.degree(self.deptree,wordindex)
        return D

    def l_context_feats(self):
        wordindex = self.index + 1
        headindex = dep_head_of(self.deptree,wordindex)
        D = {}
        D["l_brown_bag"] = "_"
        D["l_context_embed"] = "_"

        return D



    def featurize(self):
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        D.update(self.c_positional_feats())
        D.update(self.d_frequency_feats())
        D.update(self.e_morphological_feats())
        D.update(self.f_prob_in_context_feats())
        D.update(self.g_char_complexity_feats())
        D.update(self.h_brownpath_feats())
        D.update(self.i_browncluster_feats())
        D.update(self.j_embedding_feats())
        D.update(self.k_dependency_feats())
        D.update(self.l_context_feats())

        return D

    def featurize_lightweight(self): ## smaller set of features used for dev
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        D.update(self.c_positional_feats())
        D.update(self.d_frequency_feats())
        D.update(self.f_prob_in_context_feats())
        D.update(self.g_char_complexity_feats())
        D.update(self.k_dependency_feats())
        D.update(self.l_context_feats())
        return D

    def baselinefeatures(self):
        D = {}
        D.update(self.a_simple_feats_lexicalized())
        return D

def prettyprintweights(linearmodel,vectorizer):
   for name, value in zip(vectorizer.feature_names_, linearmodel.coef_[0]):
       print("\t".join([name,str(value)]))

def collect_features(data):
    labels = []
    featuredicts = []
    
    print("Collecting features...")
    count=0
    for s in readSentences(data):
       print("\r"+str(count), end="")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordInContext(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"])
                featuredicts.append(w.featurize())
                labels.append(w.label)
    print()
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    return features, labels, vec

def crossval(features, labels, vec):
    maxent = LogisticRegression(penalty='l1')
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    coeffcounter = Counter(vec.feature_names_)
    negfeats = set(vec.feature_names_)
    posfeats = set(vec.feature_names_)

    scores = defaultdict(list)
    TotalCoeffCounter = Counter()

    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels[TestIndices]

        maxent.fit(TrainX_i,Trainy_i)
        ypred_i = maxent.predict(TestX_i)
        coeffs_i = list(maxent.coef_[0])
        coeffcounter_i = Counter(vec.feature_names_)
        for value,name in zip(coeffs_i,vec.feature_names_):
            coeffcounter_i[name] = value

        acc = accuracy_score(ypred_i, Testy_i)
        pre = precision_score(ypred_i, Testy_i)
        rec = recall_score(ypred_i, Testy_i)
        # shared task uses f1 of *accuracy* and recall!
        f1 = 2 * acc * rec / (acc + rec)

        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        scores["Precision"].append(pre)
        scores["Recall"].append(rec)

        posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))

    print("Pervasive positive: ", posfeats)
    print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")

    maxent.fit(features,labels) # fit on everything

    coeffs_total = list(maxent.coef_[0])
    for value,name in zip(coeffs_total,vec.feature_names_):
            TotalCoeffCounter[name] = value

    for (key,value) in TotalCoeffCounter.most_common()[:20]:
        print(key,value)
    print("---")
    for (key,value) in TotalCoeffCounter.most_common()[-20:]:
        print(key,value)
    print("lowest coeff:",coeffcounter.most_common()[-1])
    print("highest coeff",coeffcounter.most_common()[0])

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defaultdata = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=defaultdata)
    args = parser.parse_args()

    features, labels, vec = collect_features(args.train)
    crossval(features, labels, vec)

    sys.exit(0)


if __name__ == "__main__":
    main()
