import os, gzip, lm 

def read_brown_clusters(src):
    print('\tReading brown clusters...', end='')
    d={}
    infile=open(src, 'r')
    c=[]; ch={}
    for line in infile.readlines():
        data=line.strip().split('\t')
        d[data[1]]=data[0]
        if data[0] not in c:
            c.append(data[0])
    print('Done!')
    #calculation of heights
    print('\tCalculating brown distances...', end='')
    c.sort(key=len, reverse=True)
    for x in c:
        if x[:-1] not in ch.keys():
            ch[x]=1
        else:
            ch[x]=ch[x[:-1]]+1
    print('Done!')
    return d, ch

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
    
