#!/bin/bash
# Set this to the location on the Backend where "examples/word_language_model" from Step 1
# in "Getting Started" is located
HOMEDIR=/home/chipn/dev/pytorch-examples/word_language_model # CHANGE THIS to the path to Pytorch examples' repository
cd $HOMEDIR
mkdir -p tuning_runs
LOGDIR=`mktemp -d -p "tuning_runs"`
python main.py --cuda --epochs 6 --tied --save $RANDOM  "$@"