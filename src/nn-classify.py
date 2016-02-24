import sys, os
import pickle
import numpy as np
import argparse
import feats_and_classify_py2 as feats_and_classify
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

def combine_data(train_path, test_path, out_path):
    cutoff = 0
    with open(out_path, 'w') as outFile:
        with open(train_path, 'r') as tr:
            lines = tr.readlines()
            for l in lines:
                if (len(l.strip()) > 1) and l.strip()[-1] in ['0', '1']:
                    cutoff += 1
                outFile.write(l)
        with open(test_path, 'r') as te:
            outFile.write(te.read())
    return cutoff

def main():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    data = scriptdir+'/../data/cwi_training/cwi_training.txt.lbl.conll'
    testdata = scriptdir+'/../data/cwi_testing/cwi_testing.gold.txt.lbl.conll'
    pickled_data = scriptdir+'/../data.pickle'
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', '-t', type=float, help='Threshold for predicting 0/1. If not specified, the optimal threshold will first be computed as the median of all CV splits. May take a while.')
    parser.add_argument('--iterations', '-i', type=int, default=50, help='Training iterations.')
    parser.add_argument('--hidden-layers', '-l', dest='layers', required=True, type=int, nargs='+', help='List of layer sizes')
    parser.add_argument('--cv-splits', '-c', dest='splits', type=int, help='No. of crossvalidation splits. If not specified, no CV will be performed.')
    parser.add_argument('--data', '-d', default=data, help='Features and labels')
    parser.add_argument('--testdata', '-y', default=testdata,  help='Test data (not needed for crossval).')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Print average loss at every training iteration.')
    parser.add_argument('--output', '-o', help="Output file")
    parser.add_argument('--features', '-f', dest='features', default=[], type=str, nargs='+', help='List of feature types')

    args = parser.parse_args()
    # X, y = load_pickled(args.data)
    combined_data = 'X_y_all.txt'
    cutoff = combine_data(args.data, args.testdata, combined_data)
    X, y, _ = feats_and_classify.collect_features(combined_data, True, args.features)
    X_tr = X[:cutoff]
    y_tr = y[:cutoff]
    X_te = X[cutoff:]
    y_te = y[cutoff:]
    conf = NeuralNetConfig(X=X, y=y, layers=args.layers, iterations=args.iterations, verbose=args.verbose)

    if args.splits:
        if args.threshold:
            crossval(X_tr,y_tr,args.splits, conf, t=args.threshold)
        else:
            # compute optimal threshold for each CV split
            print '### Computing optimal threshold... '
            ts = crossval(X_tr,y_tr,args.splits, conf)
            avg = np.average(ts)
            med = np.median(ts)
            print '\nThresholds for crossval splits:', ts
            print 'Mean threshold', avg
            print 'Median threshold', med
            print 'Threshold st.dev.', np.std(ts)
            # Run CV with fixed avg/median threshold
            print '\n\n### Running with avg. threshold... '
            crossval(X_tr,y_tr,args.splits, conf, t=avg)
            print '\n\n### Running with med. threshold... '
            crossval(X_tr,y_tr,args.splits, conf, t=med)
    else:
        
        nn = NN(conf)
        nn.train(X_tr,y_tr,args.iterations)
        if args.testdata:
            # X_test, y_test = load_pickled(args.testdata)
            pred = nn.get_output(X_te)
            if args.output:
                with open(args.output, 'w') as of:
                    for p in pred:
                        of.write('%f\n'%p)
            t, res = nn.test(X_te,y_te,args.threshold)
            resout = "G: %f, R: %f, A: %f, P: %f\n"%res
            sys.stderr.write('%s %f\n'%(' '.join(args.features), t))
            sys.stderr.write(resout)

if __name__ == '__main__':
    main()
