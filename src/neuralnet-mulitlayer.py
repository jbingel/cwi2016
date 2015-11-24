from pycnn import *
import pickle


with open(sys.argv[1], 'rb') as pickled_features:
    features = pickle.load(pickled_features)
with open(sys.argv[2], 'rb') as pickled_labels:
    labels = pickle.load(pickled_labels)

#features = [feats[:100] for feats in features]

NUM_LABELS=1
INPUT_DIM=len(features[0])
HIDDEN1_DIM = int(sys.argv[3])
HIDDEN2_DIM = int(sys.argv[4])

splitratio = 0.7
split = int(splitratio * len(features))
features_train = features[:split]
labels_train = labels[:split]
features_test = features[split:]
labels_test = labels[split:]

# define the parameters
m = Model()
m.add_parameters("U", (HIDDEN1_DIM,INPUT_DIM))
m.add_parameters("V", (HIDDEN2_DIM,HIDDEN1_DIM))
m.add_parameters("W", (NUM_LABELS,HIDDEN2_DIM))
m.add_parameters("b", (HIDDEN1_DIM))
m.add_parameters("c", (HIDDEN2_DIM))

# renew the computation graph
renew_cg()

# add the parameters to the graph
U = parameter(m["U"])
V = parameter(m["V"])
W = parameter(m["W"])
b = parameter(m["b"])
c = parameter(m["c"])

# create the network
x = vecInput(INPUT_DIM) 	
output = logistic(W*logistic(V*(logistic((U*x)+b))+c))
# define the loss with respect to an output y.
y = scalarInput(0) # this will hold the correct answer
loss = binary_log_loss(output, y)
# train the network
trainer = SimpleSGDTrainer(m)

if not len(features) == len(labels):
    print("Error: No of features does not match no of labels!")
    sys.exit(1)
total_loss = 0
seen_instances = 0
for f, l in zip(features_train, labels_train):
    x.set(f)
    y.set(l) 
    seen_instances += 1
    total_loss += loss.value()
    loss.backward()
    trainer.update()
    if (seen_instances > 1 and seen_instances % 100 == 0):
	print "average loss is:",total_loss / seen_instances

tp, fp, tn, fn = 0, 0, 0, 0
seen_instances = 0
for feats, lbl in zip(features_test, labels_test):
    x.set(feats)
    pred = 0 if (output.value() < 0.5) else 1
    seen_instances += 1
    #print(lbl,pred)
    if lbl == pred:
	if lbl == 1:
	    tp += 1
	else:
	    tn += 1
    else:
	if lbl == 1:
	    fn += 1
	else:
	    fp += 1
    if (seen_instances > 1 and seen_instances % 20 == 0):
	print pred, lbl
r = tp/float(tp+fn) if (tp+fn > 0) else 0.0
p = tp/float(tp+fp) if (tp+fp > 0) else 0.0
f1 = 2*p*r/(p+r) if (p+r > 0) else 0.0
print("R: ", r)
print("P: ", p)
print("F: ", f1)


