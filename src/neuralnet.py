from pycnn import *
import numpy as np
import pickle
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

class NeuralNet:

    def __init__(self, conf):
        renew_cg()
        self.model = Model()
        self.layers = conf.layers
        self._init_layers(conf.layers)
        self.x = vecInput(conf.layers[0])
        self.y = scalarInput(0) # this will hold the correct answer
	#y = vecInput(NUM_LABELS) # use for multiclass classification! also requires mapping all y to vec
        self.output = self._forward_function(conf.layers)
	self.loss = binary_log_loss(self.output, self.y)
        self.trainer = SimpleSGDTrainer(self.model)
        self.conf = conf

    def _init_layers(self, layers):
        for l in range(1, len(layers)):
            self.model.add_parameters('W'+str(l), (layers[l], layers[l-1]))
            self.model.add_parameters('b'+str(l), (layers[l]))

    def _forward_function(self, layers):
        # Input layer
        W = parameter(self.model['W1'])
        b = parameter(self.model['b1'])
        output = tanh(W * self.x) + b
        
        # Hidden layers
        l = 1
        for l in range(2, len(layers)-1):
            W = parameter(self.model['W'+str(l)])
            b = parameter(self.model['b'+str(l)])
            output = logistic(W * output + b)
       
        # Output layer 
	W = parameter(self.model['W'+str(l+1)])
	b = parameter(self.model['b'+str(l+1)])
	output = logistic(W * output + b) 
        #use softmax for several output neurons! (multiclass) 	
        #output = softmax(W * output + b) 
        return output
 
    def train(self, X_train, y_train, iterations=None):
        if iterations is None:
            iterations = self.conf.iterations
        m = len(X_train)
	for i in range(iterations):
            if self.conf.verbose: 
	        print "Iteration: "+str(i+1)
	    total_loss = 0
	    for f, l in zip(X_train, y_train):
		self.x.set(f)
		self.y.set(l) 
		total_loss += self.loss.value()
		self.loss.forward()
		self.loss.backward()
		self.trainer.update()
            self.trainer.update_epoch()
            if self.conf.verbose: 
	        print "average loss is:",total_loss/m
    
    
    def get_output(self, X):
        predictions = []
        for x in X:
            self.x.set(x)
            out = self.output.value()
            predictions.append(out)
        return predictions


    def predict_for_threshold(self, X, t):
        scores = self.get_output(X)
        return [1 if score > t else 0 for score in scores]


    def optimize_threshold(self, preds, gold):
        best_t = 0
        best_f1 = -1
        t_results = {}
        for thelp in range(1000):
	    t=thelp/1000.0
	    t_results[t] = self.eval_for_threshold(preds, gold, t)
	    f1 = t_results[t][0]
	    if f1 > best_f1:
	        best_t = t
	        best_f1 = f1
        return best_t, t_results[best_t]       


    def evaluate(self, pred, gold):
        r = recall_score(gold, pred)
        p = precision_score(gold, pred)
        a = accuracy_score(gold, pred)
        #TODO change for other tasks, this definition is only for Semeval 2016 task 11
        f1 = 2*a*r/(a+r) if (a+r > 0) else 0.0
        return (f1, r, a, p)   


    def eval_for_threshold(self, scores, gold, t):
        pred = [0 if (score < t) else 1 for score in scores]
        return self.evaluate(pred, gold)


    def test(self, X_test, y_test, t=None):
        if t:
            preds = self.predict_for_threshold(X_test, t)
            results = self.evaluate(preds, y_test)
        else:
            scores = self.get_output(X_test)
            t, results = self.optimize_threshold(scores, y_test)
        return t, results


class NeuralNetConfig:

    def __init__(self,layers=[1], iterations=1, X=[[1]], y=[1], verbose=False):
        self.layers = [len(X[0])] + layers + [1]
        self.iterations = iterations
        self.X = X
        self.y = y
        self.verbose = verbose


    def set_X(self,X):
        self.X = X

    def set_y(self,y):
        self.y = y


def oneHotLabels(labels):
    n = len(set(labels))
    ys = []
    for l in labels:
	y = np.zeros(n)
	y[l] = 1
	ys.append(y)
    return ys

