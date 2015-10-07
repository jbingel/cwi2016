#!/usr/bin/env bash
#
# Runs necessary processors in Stanford CoreNLP and outputs ConLL format

OS=`uname`
# Macs (BSD) don't support readlink -e
if [ "$OS" == "Darwin" ]; then
  scriptdir=`dirname $0`
else
  scriptpath=$(readlink -e "$0") || scriptpath=$0
  scriptdir=$(dirname "$scriptpath")
fi

if [[ "$STANFORD" == "" ]]; then
    echo "Environment variable STANFORD not set. Set to location of Stanford CoreNLP home dir.";
    exit
fi

if [ "$#" -lt 1 ]; then
    echo "Usage: ./preprocess.sh file(s)"
fi

FILES=$@
for f in $FILES
do
        cut -f1 $f | uniq > tmp; 
	java -cp "$STANFORD/*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -tokenized -annotators tokenize,ssplit,pos,lemma,ner,depparse -tokenize.whitespace -ssplit.eolonly -file tmp -outputFormat "conll" ;
        rm tmp; 
        mv tmp.conll $f.conll;
        python $scriptdir/generate_conll.py $f $f.conll;
done
