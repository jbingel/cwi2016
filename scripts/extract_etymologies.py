import sys

etymwn_loc = sys.argv[1]
targets_loc = sys.argv[2]

ancestry = []

def retrieve_etymology(word, lang="eng"):
    etymwn = open(etymwn_loc)
    global ancestry
    #ancestry = [(word, lang)]
    ancestry.append((word, lang))
    for entry in etymwn:
        if entry.startswith(lang+": "+word+"\t"):
            anc_lang, anc_word = entry.strip().split("\t")[2].split(": ") # e.g. 'grc: logos'
            if anc_word.startswith("-") or anc_word.endswith("-"): #ignore suffix 'etymologies'
                continue
            if (anc_word, anc_lang) in ancestry: # prevent cycles
                continue
            #ancestry.extend(retrieve_etymology(anc_word, anc_lang))
            retrieve_etymology(anc_word, anc_lang)
            break
    etymwn.close()
    #return ancestry

def prettyprint(ancestry):
    print(" --> ".join([lang+":"+word for word, lang in ancestry]))

if __name__ == "__main__":
    targets = open(targets_loc)
    #global ancestry
    for line in targets:
         if line.startswith("eng:"):
             ancestry = []
             lang, word = line.strip().split(": ")
             #ancestry = retrieve_etymology(word, lang)
             retrieve_etymology(word, lang)
             prettyprint(ancestry)
