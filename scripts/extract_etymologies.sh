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

ETYMWN=$scriptdir/../data/etymwn/etymwn-ety.tsv
ETYMWN_FULL=$scriptdir/../data/etymwn/etymwn.tsv
ETYMWN_TARGETS=$scriptdir/../data/etymwn/etymwn-targets.tsv

grep "rel:etymology" $ETYMWN_FULL > $ETYMWN
cut -f1 $ETYMWN | sort | uniq > $ETYMWN_TARGETS

echo "Extracting etymology..."
python $scriptdir/extract_etymologies.py $ETYMWN $ETYMWN_TARGETS > $scriptdir/../data/etymologies.txt
