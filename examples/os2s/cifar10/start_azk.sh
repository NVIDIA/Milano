#!/bin/bash
#HOMEDIR=/opt/OpenSeq2Seq
HOMEDIR=/home/okuchaiev/repos/Work/OpenSeq2Seq # CHANGE THIS to a path to your OpenSeq2Seq repository
cd $HOMEDIR
mkdir -p tuning_runs
LOGDIR=`mktemp -d -p "tuning_runs"`
python run.py --mode=train --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"
python run.py --mode=eval --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"
