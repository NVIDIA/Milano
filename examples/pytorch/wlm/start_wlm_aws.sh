#!/bin/bash
set -ex
apt-get update && apt-get install -y git libopenmpi-dev
git clone https://github.com/pytorch/examples.git
cd examples/word_language_model
ln -s /workdir workdir
model_name=$RANDOM
python main.py --cuda --epochs 3 --save workdir/$model_name  "$@"
rm workdir/$model_name