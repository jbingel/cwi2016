import os, gzip
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_edges
import pickle
from collections import defaultdict
from resources import *

def dep_head_of(sent, n):
    for u, v in sent.edges():
         if v == n:
             return u
    return None
    #return sent.predecessors(n)[0]

def dep_pathtoroot(sent,child):
    #print(child, nx.predecessor(sent,child), nx.descendants(sent,child), sent[dep_head_of(sent,child)][child]["deprel"])
    return nx.predecessor(sent,child)

def count_vowels(word):
    vowels = "aeiouAEIOU" # make this nicer, we're linguists! But a vowel is a vowel buddy ;) // was thinking of semivowels there :)
    return  sum(word.count(v) for v in vowels)

def commas_before_after(sent, idx):
    before = sum(1 for w in sent["lemma"][:idx] if w == ",")
    after = sum(1 for w in sent["lemma"][idx+1:] if w == ",")
    return before, after

def verbs_before_after(sent, idx):
    before = sum(1 for w in sent["pos"][:idx] if w.startswith("V"))
    after = sum(1 for w in sent["pos"][idx+1:] if w.startswith("V"))
    return before, after

def prob(item, level="words", corpus="wp", order=1):
    global lm_reg
    if level == "chars":
        item = " ".join([c for c in item])
    p = 0.0
    try:
        lm = lm_reg[level][corpus]
        scorefunc = {1: lm.score_ug, 2: lm.score_bg, 3: lm.score_tg}
        try:
            p = scorefunc[order](item)
        except KeyError:
            print("Warning! No entry for item '%s' in language model for level '%s' and corpus '%s'" %(item, level, corpus))
    except KeyError:
        print("Error! Could not find language model for level '%s' and corpus '%s'" %(level, corpus))
    return p

def retrieve_etymology(word, lang="eng"):
    try:
        word_etym = [edge[1] for edge in dfs_edges(etymology, lang+':'+word)]
    except KeyError:
        #print("Warning! Could not retrieve etymology for word '%s' in language '%s'" %(word, lang))
        word_etym = []
    return word_etym

def has_ancestor_in_lang(lang, word_etym):
    for ancestor in word_etym:
        if ancestor.split(':')[0] == lang:
            return True
    return False
    
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
            try:
                idx,form,lemma,pos,ne,head,deprel,label = line.split("\t")
                sent["idx"].append(int(idx))
                sent["form"].append(form)
                sent["lemma"].append(lemma)
                sent["pos"].append(pos)
                sent["ne"].append(ne)
                sent["head"].append(head)
                sent["deprel"].append(deprel)
                sent["label"].append(label)
            except:
                print(len(line.split("\t")),line.split("\t"))

    if sent["idx"]:
        yield(sent)
