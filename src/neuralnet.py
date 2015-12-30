from pycnn import *
import numpy as np
import pickle
#import featurize

def oneHotLabels(labels):
    n = len(set(labels))
    ys = []
    for l in labels:
        y = np.zeros(n)
        y[l] = 1
        ys.append(y)
    return ys

#features, labels, vec = featurize.collectFeatures(sys.argv[1])

with open(sys.argv[1], 'rb') as pickled_features:
    features = pickle.load(pickled_features)
with open(sys.argv[2], 'rb') as pickled_labels:
    labels = pickle.load(pickled_labels)

#features = features[:100]
#labels = labels[:100]

print(labels)

#NUM_LABELS=len(set(labels))
NUM_LABELS=1
#NUM_LABELS=len(set(labels))
INPUT_DIM=len(features[0])
HIDDEN_DIM1 = int(sys.argv[3])
HIDDEN_DIM2 = int(sys.argv[4])
HIDDEN_DIM3 = int(sys.argv[5])

ITER = int(sys.argv[6])

#labels = oneHotLabels(labels)
splitratio = 0.9
split = int(splitratio * len(features))
features_train = features[:split]
labels_train = labels[:split]
features_test = features[split:]
labels_test = labels[split:]

# define the parameters
m = Model()
m.add_parameters("W1", (HIDDEN_DIM1,INPUT_DIM))
m.add_parameters("W2", (HIDDEN_DIM2,HIDDEN_DIM1))
#m.add_parameters("W2", (NUM_LABELS,HIDDEN_DIM1))
m.add_parameters("W3", (HIDDEN_DIM3,HIDDEN_DIM2))
m.add_parameters("W4", (NUM_LABELS,HIDDEN_DIM3))
m.add_parameters("b1", (HIDDEN_DIM1))
m.add_parameters("b2", (HIDDEN_DIM2))
m.add_parameters("b3", (HIDDEN_DIM3))

# renew the computation graph
renew_cg()
print "Renewed CG!"
# add the parameters to the graph
W1 = parameter(m["W1"])
W2 = parameter(m["W2"])
W3 = parameter(m["W3"])
W4 = parameter(m["W4"])
b1 = parameter(m["b1"])
b2 = parameter(m["b2"])
b3 = parameter(m["b3"])

print "Creating network..."
# create the network
x = vecInput(INPUT_DIM) 	
output =  logistic(W4*  logistic(W3* logistic(W2*  logistic((W1*x) +b1) +b2) +b3))
#output =   logistic(W2*  logistic((W1*x) +b1)) 
# define the loss with respect to an output y.
y = scalarInput(0) # this will hold the correct answer
#y = vecInput(NUM_LABELS) # this will hold the correct answer
loss = binary_log_loss(output, y)
# train the network
trainer = SimpleSGDTrainer(m)

if not len(features) == len(labels):
    print("Error: No of features does not match no of labels!")
    sys.exit(1)
total_loss = 0
seen_instances = 0
print "Starting training..."

print INPUT_DIM
for i in range(ITER):
    print "Iteration: "+str(i+1)
    for f, l in zip(features_train, labels_train):
        x.set(f)
        y.set(l) 
        #print "Set in/out"
        seen_instances += 1
        total_loss += loss.value()
        #print "Computed loss"
        loss.backward()
        #print "Ran bw"
        trainer.update()
        #print "Updated"
    print "average loss is:",total_loss / seen_instances

with open('pycnn.model', 'wb') as pf:
    pickle.dump((m, output), pf, 2)
    m.save('pycnn2.mdl')

tp, fp, tn, fn = 0, 0, 0, 0
pos=0
ppred=0
res_and_gold = []
for feats, lbl in zip(features_test, labels_test):
    x.set(feats)
    res = output.value()
        
    #lbl = np.argmax(lbl)
    #pred = np.argmax(output.value())
    sys.stderr.write(str(res)+'\t'+str(lbl)+'\n')
    res_and_gold.append((res, lbl))

best_t = 0
best_f1 = -1
t_results = {}
for thelp in range(250,400):
    t=thelp/1000.0
    for res,lbl in res_and_gold:
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
    t_results[t] = (f1, r, acc, p)
    if f1 > best_f1:
        best_t = t
        best_f1 = f1

print "\nThreshold", best_t
print "\nF1", t_results[best_t][0]
print "Recall", t_results[best_t][1]
print "Accuracy", t_results[best_t][2]
print "Precision", t_results[best_t][3]

