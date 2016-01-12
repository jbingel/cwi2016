import sys, os

cwifile = open(sys.argv[1])
procfile = open(sys.argv[2])
outfiles = []

annotators=20
for a in range(annotators):
    outfiles.append(open("data/cwi_training/cwi_training_{:02d}.lbl.conll".format(a+1) , "w"))
    outfiles[a].write("# idx\tform\tlemma\tpos\tne\thead\tdeprel\tlabel\n\n")

cwiBuffer = cwifile.readline()

def cutLine(cwiLine):
    cut = cwiLine.strip().split('\t')
    try:
        sent, word, idx = cut[:3]
        votes = cut[3:]
        return sent, word, int(idx), votes
    except ValueError:
        return None

def consumeSentCwi():
    global cwiBuffer
    decisions = {}
    sent, word, idx, votes = cutLine(cwiBuffer)
    decisions[idx] = votes
    line = cwifile.readline()
    lineCut = cutLine(line)
    
    while lineCut and lineCut[0] == sent:
        idx, votes = lineCut[2:4]
        decisions[idx] = votes
        line = cwifile.readline()
        lineCut = cutLine(line)
    cwiBuffer = line
    return decisions

def consumeSentProc():
    sent = []
    line = procfile.readline().strip()
    while not line == "":
         spl = line.split()
         spl[0] = str(int(spl[0])-1)
         line = "\t".join(spl)
         sent.append(line)
         line = procfile.readline().strip()
    return sent

while True:
    sent = consumeSentProc()
    if sent == []:
        sys.exit(0)
    decisions = consumeSentCwi()
    for a in range(annotators):
        i = 0
        while i < len(sent):
            try:
                vote = decisions[i][a]
            except KeyError as e:
                vote = '-'
            outfiles[a].write(sent[i]+"\t%s\n" %vote)
            i += 1
        outfiles[a].write("\n")

for a in range(annotators):
    outfiles[a].close()
cwifile.close()

