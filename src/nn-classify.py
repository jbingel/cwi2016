import sys, os
import pickle
import numpy as np
#import feats_and_classify
from neuralnet2 import NeuralNet as NN


"""
Very hacky script, apologies for this. Will be made nicer/more flexible.
"""

with open(sys.argv[1], 'rb') as pickled_features:
    X = pickle.load(pickled_features)
with open(sys.argv[2], 'rb') as pickled_labels:
    y = pickle.load(pickled_labels)

splits = 10
iterations = 50
if len(sys.argv) > 2:
    splits = int(sys.argv[3])
if len(sys.argv) > 3:
    iterations = int(sys.argv[4])
#scriptdir = os.path.dirname(os.path.realpath(__file__))
#data = scriptdir+'/../cwi_training/cwi_training.txt.lbl.conll'

#X, y, vec = feats_and_classify.collect_features(data)

results = []
ts = []
cs = [(i*len(X)/splits, (i+1)*len(X)/splits) for i in range(splits)]
for s,e in cs:
    X_tr = [X[i] for i in range(len(X)) if i < s or i >= e]
    X_te = [X[i] for i in range(len(X)) if i >= s and i < e]
    y_tr = [y[i] for i in range(len(X)) if i < s or i >= e]
    y_te = [y[i] for i in range(len(X)) if i >= s and i < e]

    nn = NN([len(X[0]), 500, 100, 15, 1])
    nn.train(X_tr, y_tr, iterations)
    best_t, res = nn.test(X_te, y_te, t=0.1881)
    ts.append(best_t)
    results.append(res)

f1s = [res[0] for res in results]
rec = [res[1] for res in results]
acc = [res[2] for res in results]
pre = [res[3] for res in results]

print '\nF1', np.average(f1s), np.std(f1s)
print 'Rec', np.average(rec), np.std(rec)
print 'Acc', np.average(acc), np.std(acc)
print 'Pre', np.average(pre), np.std(pre)

print ts
print np.average(ts), np.std(ts)
