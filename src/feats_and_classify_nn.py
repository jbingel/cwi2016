import os, sys
from sklearn.linear_model.logistic import LogisticRegression
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

    parser.add_argument('--cv-splits', '-c', type=int, help='No of CV splits')
    parser.add_argument('--iterations', '-i', type=int, default=50, help='No of iterations')
    parser.add_argument('--hidden-layers', '-l', dest='layers', required=True, type=int, nargs='+', help='List of layer sizes.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true',  help='Print avg. loss at every iteration.')
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

    X, _, v = feats_and_classify_py2.collect_features(args.parsed_file)

    # train for every annotator...
    for vote_threshold in range(1,5):
        y_current_tr = feats_and_classify_py2.collect_labels_positive_threshold(args.all_annotations_file, vote_threshold)
        print("Training, setting positive labels for examples with at least {} positive votes. ".format(vote_threshold))
        print("Training data has {} positive labels out of {}".format(sum(y_current_tr), len(y_current_tr)))
        f1_row = [] # holds 4-tuples of (f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med) f.e. tr/te
        t_row  = [] # holds 2-tuples of (t_avg, t_med) f.e. tr/te
        f1_matrix.append(f1_row)
        t_matrix.append(t_row)
        
        conf = NeuralNetConfig(X=X, y=y_current_tr, layers=args.layers, iterations=args.iterations, verbose=args.verbose)
        print("Using neural network models with {} hidden layers of sizes {}".format(len(args.layers), args.layers))
        # optimize t for every annotator (except training annotator), yields avg/med t 
        #for idx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
        # 02, 09, 17 are the annotators with the least/average/most positive votes
        for idx in "02 09 17".split(" "):
            print("  Testing on annotator "+idx)
            #current_single_ann = scriptdir+"/../data/cwi_training/cwi_training_"+idx+".lbl.conll"

            #y_current_te = feats_and_classify_py2.collect_labels(current_single_ann)
            y_current_te = feats_and_classify_py2.collect_labels(args.all_annotations_file, int(idx)-1)
            current_label_list.append(y_current_te)
             
            thresholds, scores = getBestThresholds(X, y_current_tr, y_current_te, conf)
            t_avg = np.average(thresholds)
            t_med = np.median(thresholds)
            t_row.append((t_avg, t_med))
            f1_avg = np.average([score[0] for score in scores])
            f1_std = np.std([score[0] for score in scores])
            print("Avg. F1 for test annotator {}: {} (+/- {})".format(idx, f1_avg, f1_std))
        # calculate avg of avg t's, avg of med t's, ... for the current training annotator
        t_avg_avg = np.average([t[0] for t in t_row]) 
        t_avg_med = np.average([t[1] for t in t_row]) 
        t_med_avg =  np.median([t[0] for t in t_row]) 
        t_med_med =  np.median([t[1] for t in t_row]) 
        t_final.append((t_avg_avg, t_avg_med, t_med_avg, t_med_med))

        print("Computed optimal t's... Now running a new phase of CV experiments with these t's on test annotators.")
 
        #for idx in "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20".split(" "):
        # 02, 09, 17 are the annotators with the least/average/most positive votes
        for idx in "02 09 17".split(" "):
            f1_avg_avg = 0
            f1_med_avg = 0
            f1_avg_med = 0
            f1_med_med = 0
 
            y_current_te = feats_and_classify_py2.collect_labels(args.all_annotations_file, int(idx)-1)
            pos = sum(y_current_te)
            print("Testing globally optimal t's for annotator {} ({} positives)".format(idx, pos))
            
            #f1_avg_avg = cvWithThreshold(X, y_current_tr, y_current_te, t_avg_avg, args.regularization)['F1'][0]
            f1_avg_med = cvWithThreshold(conf, X, y_current_tr, y_current_te, t_avg_med)[0]
            #f1_med_avg = cvWithThreshold(X, y_current_tr, y_current_te, t_med_avg, args.regularization)['F1'][0]
            #f1_med_med = cvWithThreshold(X, y_current_tr, y_current_te, t_med_med, args.regularization)['F1'][0]
            print("F1 for test annotator {}: {}".format(idx, f1_avg_med))
           
            f1_row.append((f1_avg_avg, f1_avg_med, f1_med_avg, f1_med_med))

        f1_final.append(tuple(map(np.average, zip(*f1_row))))
        print(tuple(map(np.average, zip(*f1_row))))
    print(f1_final)
    # get the index (NB: array index!) of the max avg/med F1 (i.e. computed on avg/med threshold)
    best_vote_threshold_avg_avg = np.argmax([f1[0] for f1 in f1_final])
    best_vote_threshold_avg_med = np.argmax([f1[1] for f1 in f1_final])
    best_vote_threshold_med_avg = np.argmax([f1[2] for f1 in f1_final])
    best_vote_threshold_med_med = np.argmax([f1[3] for f1 in f1_final])

    print(t_final)

    sys.exit(0)


if __name__ == "__main__":
    main()
