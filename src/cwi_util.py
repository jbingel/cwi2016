import os, gzip, lm
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_edges
import pickle
from collections import defaultdict

def dep_head_of(sent, n):
    for u, v in sent.edges():
         if v == n:
             return u
    return None
    #return sent.predecessors(n)[0]

# def dep_pathtoroot(sent, child):
#     path = []
#     newhead = dep_head_of(sent, child)
#     while newhead:
#         path.append(newhead)
#         newhead = dep_head_of(sent, newhead)
#     return path

def dep_pathtoroot(sent,child):
    #print(child, nx.predecessor(sent,child), nx.descendants(sent,child), sent[dep_head_of(sent,child)][child]["deprel"])
    return nx.predecessor(sent,child)



def read_brown_clusters(src, total_clusters):
    print('\tReading brown clusters...', end='')
    d={}
    infile=open(src, 'r')
    c=[]; ch={}
    total_words=0.0; total_depths=0.0; max_depth=0
    for line in infile.readlines():
        data=line.strip().split('\t')
        d[data[1]]=data[0]
        total_words+=1
        total_depths=total_depths+len(data[0])
        if len(data[0])>max_depth:
            max_depth=len(data[0])
        if data[0] not in c:
            c.append(data[0])
    print('Done!')
    #calculation of heights
    print('\tCalculating brown heights...', end='')
    total_heights=0.0
    c.sort(key=len, reverse=True)
    for x in c:
        if x[:-1] not in ch.keys():
            ch[x]=1
        else:
            ch[x]=ch[x[:-1]]+1
    for word in d.keys():
        total_heights=total_heights+ch[d[word]]
    print('Done!')
    return d, ch, total_depths/total_words, total_heights/total_words, max_depth

def read_embeddings(src):
    print('\tReading embeddings...', end='')
    d={}
    infile=gzip.open(src, 'rt')
    for line in infile.readlines():
        data=line.strip().split(' ')
        d[data[0]]=data[1:]
    print('Done!')
    return d

def read_lm(src):
    return lm.LM(src)

print('\tReading language models... ', end='')
scriptdir = os.path.dirname(os.path.realpath(__file__))
lm_words_swp = read_lm(scriptdir+"/../data/langmodels/simplewiki.arpa")
lm_words_wp = read_lm(scriptdir+"/../data/langmodels/enwiki.arpa")
lm_chars_swp = read_lm(scriptdir+"/../data/langmodels/simplewiki_chars.arpa")
lm_chars_wp = read_lm(scriptdir+"/../data/langmodels/enwiki_chars.arpa")
lm_reg = {"words": {"swp": lm_words_swp, "wp": lm_words_wp},
          "chars": {"swp": lm_chars_swp, "wp": lm_chars_wp}}
print('Done!')

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

def loadEtym():
    etym_file = scriptdir+"/../data/etymwn.pickle"
    with open(etym_file, 'rb') as pickle_file:
        G = pickle.load(pickle_file)
    print(G.order())
    return G

def retrieve_etymology(word, lang="eng"):
    global G
    try:
        etymology = [edge[1] for edge in dfs_edges(G, lang+':'+word)]
    except KeyError:
        #print("Warning! Could not retrieve etymology for word '%s' in language '%s'" %(word, lang))
        etymology = []
    return etymology

def has_ancestor_in_lang(lang, etymology):
    for ancestor in etymology:
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
