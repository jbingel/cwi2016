import os, sys
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
import pickle
import numpy as np
import argparse
import feats_and_classify_py2
from neuralnet import NeuralNet as NN
from neuralnet import NeuralNetConfig

#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
 

def getBestThresholds(X, y_current_tr, y_current_te, conf):
    assert len(X) == len(y_current_tr) == len(y_current_te), 'Number of features ({}), annotator1 labels ({}) and annotator2 labels ({}) is not equal!'.format(len(X), len(y_current_tr), len(y_current_te))
    #scores = {"F1":[], "Recall":[], "Accuracy":[], "Precision":[]}
    scores = []
    thresholds=[]


    print('Finding best thresholds...')
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(y_current_tr, n_folds=10, shuffle=False, random_state=None):
        #print('\r'+str(fold), end="")
        fold+=1
        X_tr = X[TrainIndices]
        y_tr = y_current_tr[TrainIndices]

        X_te = X[TestIndices]
        y_te = y_current_te[TestIndices]

        nn = NN(conf)
        nn.train(X_tr, y_tr, conf.iterations)
        #get prediction
        best_t, score = nn.test(X_te, y_te)
        thresholds.append(best_t)

        scores.append(score)
    
    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("\n--")
    
    return np.array(thresholds), np.array(scores)

def cvAcrossThresholds(conf, X, y_current_tr,y_current, thresholds, average=True, median=False):
    f1_ave = 0
    f1_med = 0
    if average:
        print('Using average of best threshold...')
        t=thresholds.mean()
        f1_ave = cvWithThreshold(conf, X, y_current_tr,y_current, t)[0]
    if median:
        print('Using median of best threshold...')
        t=np.median(thresholds)
        f1_med=cvWithThreshold(conf, X, y_current_tr,y_current, t)[0]
    return f1_ave, f1_med

def cvWithThreshold(conf, X, y_current_tr, y_current_te, threshold):
    scores = []
    fold=1
    for TrainIndices, TestIndices in cross_validation.StratifiedKFold(y_current_tr, n_folds=10, shuffle=False, random_state=None):
        #print('\r'+str(fold), end="")
        fold+=1
        X_tr = X[TrainIndices]
        y_tr = y_current_tr[TrainIndices]

        X_te = X[TestIndices]
        y_te = y_current_te[TestIndices]

        nn = NN(conf)
        nn.train(X_tr, y_tr, conf.iterations)
        _, score = nn.test(X_te, y_te)

        scores.append(score)
    
    print("\n--")
    f1  = np.mean([s[0] for s in scores])
    r   = np.mean([s[1] for s in scores])
    acc = np.mean([s[2] for s in scores])
    p   = np.mean([s[3] for s in scores])

    return f1, r, acc, p

def get_args():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    default_data_parsed = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    default_data_allannotations = scriptdir+"/../data/cwi_training/cwi_training_allannotations.txt"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016--optimisation of threshhold")
    parser.add_argument('--all_annotations_file', help="parsed-and-label input format", default=default_data_allannotations)
    parser.add_argument('--parsed_file', help="parsed-and-label input format", default=default_data_parsed)
    parser.add_argument('--threshold_matrix_file', help="location/name of the threshold matrix", default='annotator_threshold_matrix')
    parser.add_argument('--regularization', help="regularizer, may be l1 or l2", default='l2')

    parser.add_argument('--cv-splits', '-c', type=int, default=10, dest='splits', help='No of CV splits')
    parser.add_argument('--iterations', '-i', type=int, default=50, help='No of iterations')
    parser.add_argument('--hidden-layers', '-l', dest='layers', required=True, type=int, nargs='+', help='List of layer sizes.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true',  help='Print avg. loss at every iteration.')
    parser.add_argument('--predict', '-p', dest='predictfile', help="Prediction file")
    parser.add_argument('--output', '-o', help="Output file", required=True)
    parser.add_argument('--threshold', '-t', type=float, help="Output threshold", required=True)

    return parser.parse_args()


def main():
    args = get_args()
    # f1_matrix holds for every training annotator: the list of tuples of 
    # avg/med f1_row based on avg/med threshold
    f1_matrix = []
    # holds for every training annotator: the list of tuples of avg/med threshold
    t_matrix = []
    current_label_list = []
    
    f1_final = [] # holds 4-tuples of avgs over (f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med) f.e. tr 
    t_final  = [] # holds 4-tuples of (t_avg_avg, t_avg_med, t_med_avg, t_med_med) f.e. tr

    #X_tr, _, v = feats_and_classify_py2.collect_features(args.parsed_file)
    with open('X_train.pickle', 'rb') as pf:
        X_tr = pickle.load(pf)
    with open('X_test.pickle', 'rb') as pf:
        X_te = pickle.load(pf)
    y_tr = feats_and_classify_py2.collect_labels_positive_threshold(args.all_annotations_file, 1)

    #X_out, _, _ = feats_and_classify_py2.collect_features(args.predictfile)
    # filter for targets
    #X_out = [x for x in X_out if not x.label == '?']

    conf = NeuralNetConfig(X=X_tr, y=y_tr, layers=args.layers, iterations=args.iterations, verbose=args.verbose)
    
    nn = NN(conf)
    nn.train(X_tr, y_tr)
    preds = nn.predict_for_threshold(X_te, 0.2444)
    with open(args.output, 'w') as outfile:
        for p in preds:
            #print(p)
            outfile.write(str(p))
            outfile.write('\n')
    sys.exit(0)


if __name__ == "__main__":
    main()
