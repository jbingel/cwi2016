PYTHON=/home/joachim/anaconda3/bin/python

touch sample_weightings_eval
for w in "uniform" "linear" "tf_idf" "log_and_mode" "inverse_class_relevance" "log_and_max"
do
echo $w >> sample_weightings_eval
$PYTHON ../src/feats_and_classify_weighted.py --instance_weighting $w >> sample_weightings_eval
done
