import argparse
from scipy.stats import pearsonr
parser = argparse.ArgumentParser(description="")
parser.add_argument('--file1')
parser.add_argument('--file2')
args = parser.parse_args()

pred1= [float(x) for x in open(args.file1).read().strip().split(" ")]
pred2= [float(x) for x in open(args.file2).read().strip().split(" ")]
print(pearsonr(pred1,pred2))