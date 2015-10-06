def count_vowels(word):
    vowels = "aeiouAEIOU" # make this nicer, we're linguists!
    return  sum(word.count(v) for v in vowels)

def commas_before_after(sent, idx):
    before = sum(1 for w in sent[:idx] if w == ",")
    after = sum(1 for w in sent[idx+1:] if w == ",")
    return before, after

"""
TODO implement get_pos()
def verbs_before_after(sent, idx):
    before = sum(1 for w in sent[:idx] if get_pos(w).startswith("V"))
    after = sum(1 for w in sent[idx+1:] if get_pos(w).startswith("V"))
    return before, after
"""

def read_lm(src):
    d = {}
    for line in open(src):
        line = line.strip()
        if line.startswith("#") or line == "":
            continue
        item, count = line.split("\t")
        d[item] = float(count)
    return d

lm_words_swp = read_lm("../data/langmodels/words.swp.lm")
lm_words_wp = read_lm("../data/langmodels/words.wp.lm")
lm_chars_swp = read_lm("../data/langmodels/chars.swp.lm")
lm_chars_wp = read_lm("../data/langmodels/chars.wp.lm")
lm_reg = {"words": {"swp": lm_words_swp, "wp": lm_words_wp},
          "chars": {"swp": lm_chars_swp, "wp": lm_chars_wp}}

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
