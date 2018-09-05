#!/bin/bash
#SBATCH --job-name=Milano
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=6000
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL

docker run --runtime=nvidia --rm --name AAA -v \
 /gpfs/fs1/datasets:/gpfs/fs1/datasets --shm-size=1G \
--ulimit memlock=-1 --ulimit stack=67108864 \
gitlab-dl.nvidia.com:5005/dgx/tensorflow:18.07-py3-stage \  # CHANGE THIS to TF container of your chose. Visit ngc.nvidia.com for NVIDIA's Tensorflow containers
/bin/bash -c 'nvidia-smi &&\
 git clone https://github.com/NVIDIA/OpenSeq2Seq.git &&\
 cd OpenSeq2Seq &&\
 pip install -r requirements.txt &&\
 ln -s  /gpfs/fs1/datasets/cifar10 data &&\
 python run.py --mode=train --logdir=here --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@" &&\
 python run.py --mode=eval --logdir=here --config_file=example_configs/image2label/cifar-nv.py --num_epochs=50 --enable_logs "$@"'
