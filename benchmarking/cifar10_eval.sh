#!/bin/bash
HOMEDIR=/opt/OpenSeq2Seq
cd $HOMEDIR
git pull
cp /data/cifar-10-binary.tar ./
tar xvf cifar-10-binary.tar
mkdir data
mv cifar-10-batches-bin data
LOGDIR=/result/cifar10
python run.py --mode=train --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"
python run.py --mode=eval --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"
