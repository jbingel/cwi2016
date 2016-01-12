import sys, os
import pickle
import numpy as np
import argparse
#import feats_and_classify
from neuralnet import NeuralNet as NN
from neuralnet import NeuralNetConfig


#TODO collecting features from feats_and_classify rather than only pre-pickled

def load_pickled(data):
    with open(data, 'rb') as pickled_data:
	X,y = pickle.load(pickled_data)
    print "Loaded data with %d instances" %(len(X))
    return X,y

def crossval(X,y,splits, conf, t=None):
    results = []
    ts = []
    m = len(X)
    cs = [(i*m/splits, (i+1)*len(X)/splits) for i in range(splits)]
    for s,e in cs:
	X_tr = [X[i] for i in range(m) if i < s or i >= e]
	X_te = [X[i] for i in range(m) if i >= s and i < e]
	y_tr = [y[i] for i in range(m) if i < s or i >= e]
	y_te = [y[i] for i in range(m) if i >= s and i < e]

	nn = NN(conf)
	nn.train(X_tr, y_tr, conf.iterations)
	best_t, res = nn.test(X_te, y_te, t)
	ts.append(best_t)
	results.append(res)

    f1s = [res[0] for res in results]
    rec = [res[1] for res in results]
    acc = [res[2] for res in results]
    pre = [res[3] for res in results]

    print '\nF1  | {:.3f}   (std {:.3f})'.format(np.average(f1s), np.std(f1s))
    print 'Rec | {:.3f}   (std {:.3f})'.format(np.average(rec), np.std(rec))
    print 'Acc | {:.3f}   (std {:.3f})'.format(np.average(acc), np.std(acc))
    print 'Pre | {:.3f}   (std {:.3f})'.format(np.average(pre), np.std(pre))

    return ts 

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    data = scriptdir+'/../cwi_training/cwi_training.txt.lbl.conll'
    pickled_data = scriptdir+'/../data.pickle'
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', '-t', type=float, help='Threshold for predicting 0/1. If not specified, the optimal threshold will first be computed as the median of all CV splits. May take a while.')
    parser.add_argument('--iterations', '-i', type=int, default=50, help='Training iterations.')
    parser.add_argument('--hidden-layers', '-l', dest='layers', required=True, type=int, nargs='+', help='List of layer sizes')
    parser.add_argument('--cv-splits', '-c', dest='splits', type=int, help='No. of crossvalidation splits. If not specified, no CV will be performed.')
    parser.add_argument('--data', '-d', default=pickled_data, help='Pickled features and labels')
    parser.add_argument('--testdata', '-y',  help='Test data (not needed for crossval).')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Print average loss at every training iteration.')

    args = parser.parse_args()
    X, y = load_pickled(args.data)
    conf = NeuralNetConfig(X=X, y=y, layers=args.layers, iterations=args.iterations, verbose=args.verbose)

    if args.splits:
        if args.threshold:
	    crossval(X,y,args.splits, conf, t=args.threshold)
        else:
            # compute optimal threshold for each CV split
	    print '### Computing optimal threshold... '
	    ts = crossval(X,y,args.splits, conf)
	    avg = np.average(ts)
	    med = np.median(ts)
            print '\nThresholds for crossval splits:', ts
            print 'Mean threshold', avg 
            print 'Median threshold', med
            print 'Threshold st.dev.', np.std(ts)
            # Run CV with fixed avg/median threshold
	    print '\n\n### Running with avg. threshold... '
	    crossval(X,y,args.splits, conf, t=avg)
	    print '\n\n### Running with med. threshold... '
	    crossval(X,y,args.splits, conf, t=med)
    else:
        nn = NN(conf)
        nn.train(X,y,args.iterations)
        if args.testdata:
            X_test, y_test = load_pickled(args.testdata)
            nn.test(X_test,y_test,args.threshold)

if __name__ == '__main__':
    main()
