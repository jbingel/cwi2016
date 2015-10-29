import argparse
import os
from sklearn.linear_model.logistic import LogisticRegression
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *
import porter

class WordInContext:
    def __init__(self,sentence,index,word,lemma,pos,namedentity,label,heads,deprels):
        self.sentence = sentence #sentence is a list of forms
        self.word = word
        self.index = int(index)
        self.label = int(label)
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
        etymology = retrieve_etymology(self.word)
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
        trigramProb = prob(self.word, level="chars")
        trigramProbSimple = prob(self.word, level="chars", corpus="swp")
        D["g_char_trigram_prob"] = trigramProb
        D["g_char_prob_ratio"] = trigramProbSimple / trigramProb
        D["g_vowels_ratio"] = float(count_vowels(self.word)) / len(self.word)
        return D 
    
    def h_abstract_feats(self):
        D={}
        #brown cluster path feature
        global brownclusters, embeddings
        if self.word in brownclusters:
            bc = brownclusters[self.word]
            for i in range(1,len(bc)):
                D["h_cluster_"+bc[0:i] ]=1
        
            #brown cluster height=general/depth=fringiness
            D["h_cluster_height"]=len(bc)
            D["h_cluster_depth"]=cluster_heights[bc]
        
        #word embedding
        if self.word in embeddings.keys():
            emb=embeddings[self.word]
            for d in range(len(emb)):
                D["h_embed_"+str(d)]=emb[d]

        #TODO: (1) re-embedded embeddings, (2) fringiness of embedding 
        return D


    def featurize(self):
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        D.update(self.c_positional_feats())
        #D.update(self.d_frequency_feats())
        D.update(self.e_morphological_feats())
        D.update(self.f_prob_in_context_feats())
        D.update(self.g_char_complexity_feats())
        D.update(self.h_abstract_feats())
        return D

def prettyprintweights(linearmodel,vectorizer):
   for name, value in zip(vectorizer.feature_names_, linearmodel.coef_[0]):
       print("\t".join([name,str(value)]))


def readSentences(infile):
    sent = defaultdict(list)
    #0    In    in    IN    O    4    case    -

    for line in open(infile).readlines():
        line = line.strip()
        if not line:
            yield(sent)
            sent = defaultdict(list)
        elif line.startswith("#"):
            pass
        else:
            idx,form,lemma,pos,ne,head,deprel,label = line.split("\t")
            sent["idx"].append(int(idx))
            sent["form"].append(form)
            sent["lemma"].append(lemma)
            sent["pos"].append(pos)
            sent["ne"].append(ne)
            sent["head"].append(head)
            sent["deprel"].append(deprel)
            sent["label"].append(label)

    if sent["idx"]:
        yield(sent)



brownclusters, cluster_heights=read_brown_clusters('/coastal/brown_clusters/rcv1.64M-c10240-p1.paths')
embeddings=read_embeddings('/coastal/mono_embeddings/glove.6B.300d.txt.gz')

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defaultdata = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
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
                featuredicts.append(w.featurize())
                labels.append(w.label)
    print()
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()

    maxent = LogisticRegression()
    maxent.fit(features,labels)
    #prettyprintweights(maxent,vec)
    coeffs = list(maxent.coef_[0])
    lowest = min(coeffs)
    highest = max(coeffs)
    print("--")
    print("lowest coeff:",lowest, vec.feature_names_[coeffs.index(lowest)])
    #print(lowest, vec.feature_names_[coeffs.index(lowest)])
    print("highest coeff",highest, vec.feature_names_[coeffs.index(highest)])




if __name__ == "__main__":
    main()
