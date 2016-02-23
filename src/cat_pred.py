import os, sys, math
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from cwi_util import *
import porter
import numpy as np
#from resources import *
<<<<<<< HEAD
import argparse, feats_and_classify_py2
=======
import argparse
import feats_and_classify
from operator import itemgetter
import pickle
>>>>>>> 020aa3a889ff76800a53adddf5c7d618dd548563



def	predictTestSet():
	#generate training features and labels
	trainfile='/home/natschluter/GroupAlgorithms/cwi2016/data/cwi_training/cwi_training_cat.lbl.conll'
	trainfeatures, trainlabels, vec = feats_and_classify_py2.collect_features(trainfile)
	#generate training+test features
	
	bothfiles='/home/natschluter/GroupAlgorithms/cwi2016/data/train_and_test1.conll'
	bothfeatures, bothlabels, bothvec = feats_and_classify_py2.collect_features(bothfiles)
	thresholds_med=np.median(np.array([ 0.145,  0.85,   0.12,   0.657,  0.71,   0.824,  0.506,  0.461,  0.662,  0.888]))
	
	TrainX=bothfeatures[np.array(range(len(trainfeatures)))]
	TrainY=bothlabels[np.array(range(len(trainlabels)))]
	TestX=bothfeatures[np.array(range(len(trainfeatures),len(bothfeatures)))]
	maxent = LogisticRegression(penalty='l2')
	print('training...')
	maxent.fit(TrainX,TrainY)
	print('predicting...')
	ypred_probs=maxent.predict_proba(TestX)
<<<<<<< HEAD
	#ypred=[1 if pair[1]>=thresholds_med else 0 for pair in ypred_probs]
        ypred=ypred_probs
	outfile=open('cat_predictions.txt','w')
=======
	pickle.dump(ypred_probs, open('ypred_probs.p', 'wb'))
		
	
	allprobs=sorted(ypred_probs, reverse=True, key=itemgetter(1))
	pickle.dump(allprobs, open('allprobs.p','wb'))
	index=int(math.floor(float(len(allprobs))/10.0))
	threshold=allprobs[index][1]
	ypred=[1 if pair[1]>=threshold else 0 for pair in ypred_probs]	
		
	print('resulting threshold for 10 percent '+str(threshold))
	outfile=open('cat_predictions_top10percent.txt','w')
	outfile.write('\n'.join([str(item) for item in ypred]))
	outfile.close()

	index=int(math.floor(float(len(allprobs))/6.67))
	threshold=allprobs[index][1]
	ypred=[1 if pair[1]>=threshold else 0 for pair in ypred_probs]	
		
	print('resulting threshold for 15 percent'+str(threshold))
	outfile=open('cat_predictions_top15percent.txt','w')
	outfile.write('\n'.join([str(item) for item in ypred]))
	outfile.close()

	index=int(math.floor(float(len(allprobs))/5.0))
	threshold=allprobs[index][1]
	ypred=[1 if pair[1]>=threshold else 0 for pair in ypred_probs]	
		
	print('resulting threshold for 20 percent'+str(threshold))
	outfile=open('cat_predictions_top20percent.txt','w')
>>>>>>> 020aa3a889ff76800a53adddf5c7d618dd548563
	outfile.write('\n'.join([str(item) for item in ypred]))
	outfile.close()
	
def main():
	predictTestSet()
	
if __name__ == "__main__":
	main()
