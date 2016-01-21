import sys, os

infile = open(sys.argv[1])
procfile = open(sys.argv[2])
outfile = open(sys.argv[1]+".lbl.conll", "w")
outfile.write("# idx\tform\tlemma\tpos\tne\thead\tdeprel\tlabel\n\n")

cwiBuffer = infile.readline()

def cutLine(cwiLine):
    try:
        sent, word, idx = cwiLine.strip().split("\t")
        return sent, word, int(idx)
    except ValueError:
        return None

def consumeSentCwi():
    global cwiBuffer
    idxs = []
    line = infile.readline()
    bufferCut = cutLine(cwiBuffer)
    idxs.append(int(bufferCut[2]))
    lineCut = cutLine(line)
    while lineCut and lineCut[0] == bufferCut[0]:
        idxs.append(int(lineCut[2]))
        line = infile.readline()
        lineCut = cutLine(line)
    cwiBuffer = line
    return idxs

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
    idxs = consumeSentCwi()
    i = 0
    while i < len(sent):
        label = '?' if i in idxs else '-'
        outfile.write(sent[i]+"\t%s\n" %label)
        i += 1
    outfile.write("\n")

infile.close()
outfile.close()

