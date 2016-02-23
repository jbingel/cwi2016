import os, sys
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble  import RandomForestClassifier
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import pickle
import numpy as np
import argparse
import feats_and_classify_py2

 


def get_args():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    default_data_parsed = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    default_data_allannotations = scriptdir+"/../data/cwi_training/cwi_training_allannotations.txt"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016--optimisation of threshhold")
    parser.add_argument('--all_annotations_file', help="parsed-and-label input format", default=default_data_allannotations)
    parser.add_argument('--parsed_file', help="parsed-and-label input format", default=default_data_parsed)
    parser.add_argument('--threshold_matrix_file', help="location/name of the threshold matrix", default='annotator_threshold_matrix')
    parser.add_argument('--regularization', help="regularizer, may be l1 or l2", default='l2')

    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true',  help='Print avg. loss at every iteration.')
    parser.add_argument('--predict', '-p', dest='predictfile', help="Prediction file")
    parser.add_argument('--gold', '-g', default='/home/joachim/proj/cwi2016/data/cwi_testing/gold.labels', help="gold labels file")
    parser.add_argument('--output', '-o', help="Output file", required=True)
    parser.add_argument('--threshold', '-t', type=float, help="Output threshold")

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
    y_te = np.array([int(l) for l in open(args.gold)])
    #X_out, _, _ = feats_and_classify_py2.collect_features(args.predictfile)
    # filter for targets
    #X_out = [x for x in X_out if not x.label == '?']

    rf = RandomForestClassifier(11)
    rf.fit(X_tr, y_tr) 
    preds = rf.predict_proba(X_te)
 
   

    with open(args.output, 'w') as outfile:
        for p in preds:
            #print(p)
            outfile.write(str(p[1]))
            outfile.write('\n')
    sys.exit(0)


if __name__ == "__main__":
    main()
