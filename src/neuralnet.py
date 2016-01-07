from pycnn import *
import numpy as np
import pickle

class NeuralNet:

    def __init__(self, layers):
	renew_cg()
        self.model = Model()
        self.layers = layers
        self._init_layers(layers)
        self.x = vecInput(layers[0])
	self.y = scalarInput(0) # this will hold the correct answer
	#y = vecInput(NUM_LABELS) # this will hold the correct answer
        self.output = self._forward_function(layers)
	self.loss = binary_log_loss(self.output, self.y)
        self.trainer = SimpleSGDTrainer(self.model)

    def _init_layers(self, layers):
        for l in range(len(layers)-1):
            self.model.add_parameters('W'+str(l+1), (layers[l+1], layers[l]))
            self.model.add_parameters('b'+str(l+1), (layers[l+1]))

    def _forward_function(self, layers):
        output = self.x 
        for l in range(1, len(layers)-1):
            W = parameter(self.model['W'+str(l)])
            b = parameter(self.model['b'+str(l)])
            output = logistic(W * output + b)
	W = parameter(self.model['W'+str(l+1)])
	output = logistic(W * output)
        return output
 
    def train(self, X_train, y_train, iterations):
	seen_instances = 0
	total_loss = 0
	for i in range(iterations):
	    print "Iteration: "+str(i+1)
	    for f, l in zip(X_train, y_train):
		self.x.set(f)
		self.y.set(l) 
		seen_instances += 1
		total_loss += self.loss.value()
		#print "Computed loss"
		self.loss.backward()
		#print "Ran bw"
		self.trainer.update()
		#print "Updated"
	    print "average loss is:",total_loss / seen_instances


    def test(self, X_test, y_test, t=None):
	res_and_gold = []
	for feats, lbl in zip(X_test, y_test):
	    self.x.set(feats)
	    res = self.output.value()
		
	    #lbl = np.argmax(lbl)
	    #pred = np.argmax(output.value())
	    #sys.stderr.write(str(res)+'\t'+str(lbl)+'\n')
	    res_and_gold.append((res, lbl))
        if t:
            results = pred_for_threshold(res_and_gold, t)
        else:
            t, results = optimize_threshold(res_and_gold)
        return t, results

def oneHotLabels(labels):
    n = len(set(labels))
    ys = []
    for l in labels:
	y = np.zeros(n)
	y[l] = 1
	ys.append(y)
    return ys

def optimize_threshold(pred_and_gold):
    best_t = 0
    best_f1 = -1
    t_results = {}
    for thelp in range(1000):
	t=thelp/1000.0
	t_results[t] = pred_for_threshold(pred_and_gold, t)
	f1 = t_results[t][0]
	if f1 > best_f1:
	    best_t = t
	    best_f1 = f1
    return best_t, t_results[best_t]       

def pred_for_threshold(pred_and_gold, t):
    tp, fp, tn, fn = 0, 0, 0, 0
    pos=0
    ppred=0
    for res,lbl in pred_and_gold:
	pred = 0 if (res < t) else 1
	if lbl == 1:
	    pos += 1
	if pred == 1:
	    ppred += 1
	if lbl == pred:
	    if pred == 1:
		tp += 1
	    else:
		tn += 1
	else:
	    if pred == 1:
		fp += 1
	    else:
		fn += 1
	r = tp/float(tp+fn) if (tp+fn > 0) else 0.0
	p = tp/float(tp+fp) if (tp+fp > 0) else 0.0
	acc = float(tp+tn)/(tp+fp+tn+fn)
	f1 = 2*acc*r/(acc+r) if (acc+r > 0) else 0.0
    return (f1, r, acc, p)   


