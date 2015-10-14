import os 

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

def count_vowels(word):
    vowels = "aeiouAEIOU" # make this nicer, we're linguists!
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
    
