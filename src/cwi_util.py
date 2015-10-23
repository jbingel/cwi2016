import os, gzip 

def read_brown_clusters(src):
    print '\tReading brown clusters...'
    d={}
    infile=open(src, 'r')
    c=[]; ch={}
    for line in infile.readlines():
        data=line.strip().split('\t')
        d[data[1]]=data[0]
        if data[0] not in c:
            c.append(data[0])
    #calculation of heights
    print '\tCalculating brown distances...'
    c.sort(key=len, reverse=True)
    for x in c:
        if x[:-1] not in ch.keys():
            ch[x]=1
        else:
            ch[x]=ch[x[:-1]]+1
    return d, ch

def read_embeddings(src):
    print '\tReading embeddings...'
    d={}
    infile=gzip.open(src, 'r')
    for line in infile.readlines():
        data=line.strip().split(' ')
        d[data[0]]=data[1:]
    return d

def read_lm(src):
    d = {}
    for line in open(src):
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        item, count = line.split("\t")
        d[item] = float(count)
    return d

scriptdir = os.path.dirname(os.path.realpath(__file__))
lm_words_swp = read_lm(scriptdir+"/../data/langmodels/words.swp.lm")
lm_words_wp = read_lm(scriptdir+"/../data/langmodels/words.wp.lm")
lm_chars_swp = read_lm(scriptdir+"/../data/langmodels/chars.swp.lm")
lm_chars_wp = read_lm(scriptdir+"/../data/langmodels/chars.wp.lm")
lm_reg = {"words": {"swp": lm_words_swp, "wp": lm_words_wp},
          "chars": {"swp": lm_chars_swp, "wp": lm_chars_wp}}
#brown_clusters=read_brown_clusters("/coastal/brown_clusters/rcv1.64M-c10240-p1.paths")
#embeddings=read_embeddings("/coastal/mono_embeddings/glove.6B.300d.txt.gz")

def count_vowels(word):
    vowels = "aeiouAEIOU" # make this nicer, we're linguists! But a vowel is a vowel buddy ;)
    return  sum(word.count(v) for v in vowels)

def commas_before_after(sent, idx):
    before = sum(1 for w in sent["lemma"][:idx] if w == ",")
    after = sum(1 for w in sent["lemma"][idx+1:] if w == ",")
    return before, after

def verbs_before_after(sent, idx):
    before = sum(1 for w in sent["pos"][:idx] if w.startswith("V"))
    after = sum(1 for w in sent["pos"][idx+1:] if w.startswith("V"))
    return before, after

def freq(item, level="words", corpus="wp"):
    global lm_reg
    d = {}
    f = 0.0
    try:
        d = lm_reg[level][corpus]
    except KeyError:
        print("Error! Could not find language model for level '%s' and corpus '%s'" %(level, corpus))
    try:
        f = d[item]
    except KeyError:
        print("Warning! No entry for item '%s' in language model for level '%s' and corpus '%s'" %(item, level, corpus))
    return f

def retrieve_etymology(word, lang="eng"):
    etym_file = scriptdir+"/../data/etymwn/etymologies.%s.txt" %word[0].lower()
    if not os.path.isfile(etym_file):
        print("Warning: Could not find etymology for word '%s' in language '%s'" %(word, lang))
        return []
    etymologies = open(etym_file)
    for line in etymologies:
        if line.startswith(lang+":"+word+" --> "):
            return [ancestor.split(":") for ancestor in line.split(" --> ")[1:]]
    return []

def has_ancestor_in_lang(lang, etymology):
    for ancestor in etymology:
        if ancestor[0] == lang:
            return True
    return False
    
