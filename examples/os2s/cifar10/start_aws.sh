#!/bin/bash
set -ex
HOMEDIR=/tmp/OpenSeq2Seq
mkdir -p /tmp/OpenSeq2Seq
apt-get update && apt-get install -y git libopenmpi-dev
git clone https://github.com/NVIDIA/OpenSeq2Seq.git $HOMEDIR
cd $HOMEDIR
ln -s /data data
LOGDIR=/tmp/logs
pip install -r requirements.txt
python run.py --mode=train --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"
python run.py --mode=eval --logdir=$LOGDIR --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"