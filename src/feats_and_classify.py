import argparse
from sklearn.linear_model.logistic import LogisticRegression
from collections import Counter
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *

class WordInContext:
    def __init__(self,sentence,word,index,label):
        self.sentence = sentence.strip().split()
        self.word = word
        self.index = int(index)
        self.label = int(label)

    def a_simple_feats(self): #
        D = {}
        D["a_form"] = self.word
        # TODO D["a_lemma"] = lemmatize(self.word)
        # TODO D["a_pos"] = get_pos(self.word)
        # TODO D["a_ne"] = is_named_entity(self.word)  # uses proper NER (Stanford or the like)
        D["a_formlength"] = len(self.word)
        return D


    def b_wordnet_feats(self): #
        D = {}
        D["b_nsynsets"] = len(wn.synsets(self.word))
        return D

    def c_positional_feats(self):
        D = {}
        D["c_relativeposition"] =  int(self.index / len(self.sentence))
        D["c_preceding_commas"] = commas_before_after(self.sentence, self.index)[0]
        D["c_following_commas"] = commas_before_after(self.sentence, self.index)[1]
        # TODO D["c_preceding_verbs"] = verbs_before_after(self.sentence, self.index)[0]
        # TODO D["c_following_verbs"] = verbs_before_after(self.sentence, self.index)[1]
        return D
        
    def d_frequency_feats(self):
        D = {}
        # TODO D["d_freq_in_swp"] = freq(self.word, "swp")
        # TODO D["d_freq_in_wp"] = freq(self.word, "wp")
        # TODO D["d_freq_ratio_swp/wp"] = freq(self.word, "swp") / freq(self.word, "wp")
        # TODO D["d_freqrank_distance_swp/wp"] = rank(self.word, "swp") - rank(self.word, "wp")  
        # TODO D["d_distributional_distance_swp/wp"] = dist(dist_vector(self.word, "swp"), dist_vector(self.word, "wp"))  # get distributional vector from background corpora, use some dist measure
        return D

    def e_morphological_feats(self):
        D = {}
        # TODO D["e_foreign_root"] = has_foreign_root(self.word)  # check wiktionary
        # TODO D["e_length_dist_lemma_form"] = len(self.word) - len(lemmatize(self.word))
        # TODO D["e_length_dist_stem_form"] = len(self.word) - len(stem(self.word))
        # TODO D["e_inflectional_morphemes_count"] = porterstemmer.reductions(self.word)  # implement interations counter in Porter Stemmer
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
        # TODO D["g_char_unigram_prob"] = char_prob(self.word, 1)   # prob() uses freq()
        # TODO D["g_char_bigram_prob"] = char_prob(self.word, 2)   # prob() uses freq()
        D["g_vowels_ratio"] = float(count_vowels(self.word)) / len(self.word)
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
