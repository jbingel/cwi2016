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
