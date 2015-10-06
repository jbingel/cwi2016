import sys, os

infile = open(sys.argv[1])
outfile = open(sys.argv[1]+".dummy.conll", "w")

curSent = ""
curSentDecisions = {}

outfile.write("# idx\tform\tlemma\tpos\tne\thead\tdeprel\tlabel\n")

for line in infile:
    sent, word, idx, label = line.strip().split("\t")
    curSentDecisions[int(idx)] = label
    if not sent == curSent:
        i = 0
        for token in curSent.split():
            lbl = curSentDecisions.get(i, "-")
            lemma = token.lower() # TODO lemmatise
            pos = "_" # TODO postag
            ne =  "_" # TODO NER
            head = 0 # TODO parse
            deprel = "_"
            outfile.write("%d\t%s\t%s\t%s\t%s\t%d\t%s\t%s\n" %(i, token, lemma, pos, ne, head, deprel, lbl))
            i += 1
        outfile.write("\n")
        curSent = sent
infile.close()
outfile.close()

