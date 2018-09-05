#!/bin/bash
#SBATCH --job-name=Milano
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=6000
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

# CHANGE THIS to TF container of your chose. Visit ngc.nvidia.com for NVIDIA's Tensorflow containers

docker run --runtime=nvidia --rm --name AAA -v \
/home/okuchaiev:/home/okuchaiev -v /mnt:/mnt --shm-size=1G \
--ulimit memlock=-1 --ulimit stack=67108864 \
gitlab-dl.nvidia.com:5005/dgx/pytorch:18.07-py3-stage \ # CHANGE THIS to Pytorch container of your choice. Visit ngc.nvidia.com for NVIDIA's Pytorch containers
/bin/bash -c 'nvidia-smi &&\
 mkdir tmp && cd tmp &&\
 git clone https://github.com/pytorch/examples.git &&\
 cd examples/word_language_model && python main.py --cuda --epochs 6 --save $RANDOM  "$@"'
