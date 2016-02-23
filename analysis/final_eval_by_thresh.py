import sys
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

def g_score(r,a):
    return (2*r*a)/(r+a)

def load(infile):
    return np.array([float(y.strip()) for y in open(infile)])
    
def binarize(ys, t):
    return np.array([1 if y>=t else 0 for y in ys])

pred = load(sys.argv[1])
pred = binarize(pred, float(sys.argv[2]))
gold = load('../data/cwi_testing/gold.labels')

r = recall_score(gold, pred)
a = accuracy_score(gold, pred)
p = precision_score(gold, pred)
f = f1_score(gold, pred)
g = g_score(r, a)

print("R: {}".format(r))
print("A: {}".format(a))
print("P: {}".format(p))
print("F: {}".format(f))
print("G: {}".format(g))
