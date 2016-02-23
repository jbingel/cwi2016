def cutLine(cwiLine):
    try:
        sent, word, idx, label = cwiLine.strip().split("\t")
        return sent, word, int(idx), label
    except ValueError:
        return None

def consumeSentCwi(infile):
    global cwiBuffer
    decisions = []
    outLines = []
    line = infile.readline()
    bufferCut = cutLine(cwiBuffer)
    outLines.append(cwiBuffer)
    decisions.append((bufferCut[2],bufferCut[3]))
    lineCut = cutLine(line)
    while lineCut and lineCut[0] == bufferCut[0]:
        decisions.append((lineCut[2],lineCut[3]))
        outLines.append(line)
        line = infile.readline()
        lineCut = cutLine(line)

    cwiBuffer = line
    return decisions, outLines

def writeOrdered(outfile, outLines):
    lines = sorted(outLines, key=lambda x: cutLine(x)[2])
    for l in lines:
        outfile.write(l)


# def checkForBug(decisions):
#     # decisions is list of tuples (idx, lbl)
#     seen_one = False
#     seen_zero = False
#     for pair in decisions:
#         if pair[1] == '0':
#             seen_zero = True
#             if seen_one:
#                 return '10'
#         elif pair[1] == '1':
#             seen_one=True
#             if seen_zero:
#                 return '01'
#
#     return '00'

infile = open("/home/joachim/Desktop/cwi_testing_annotated.txt")
outfile = open("/home/joachim/Desktop/cwi_testing_annotated.ordered.txt", "w")
cwiBuffer = infile.readline()
dec, sent = consumeSentCwi(infile)

while sent:
    writeOrdered(outfile, sent)
    print(len(sent))
    # x=checkForBug(dec)
    # d[x] = d[x]+1
    # i+=1
    # print(i,d)
    dec, sent = consumeSentCwi(infile)

