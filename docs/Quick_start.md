# Quick Start
This example will demonstrate how to optimize a simple word-level language model with Milano

## Before you start
* Make sure you have access to the Backend: It can be SLURM-managed cluster, AWS, or a machine with at least one NVIDIA GPU
* Use Python 3.5 or 3.6 and install all the requirements ``pip install -r requirements.txt``

In the example below we assume that our Backend is a machine with at least one NVIDIA GPU.

## Step 1 (Prepare model implementation and data)

Milano is framework agnostic. As such, it can be used to optimize scripts based on Pytorch, Tensorflow or any other framework.

* On the Backend machine, clone the following GitHub repository: ``https://github.com/pytorch/examples.git``
* Take a look at the [options for this language model](https://github.com/pytorch/examples/tree/master/word_language_model). We are going to tune those as an example.

**Attention:** Note, that this repository also contains it's training data.
For "real" examples you would need to make sure that training and evaluation data is also available by workers on your Backend.
For example, if you are using AWS, you might want to keep your data in a dedicated storage, or if you are using SLURM cluster, your 
data might reside on a network share. 

## Step 2 (Setup the backend)

On the Backend machine, install Azkaban solo server:
* ``git clone https://github.com/azkaban/azkaban.git``.
* ``cd azkaban; ./gradlew build installDist`` // make sure you have installed Java >=8.
* ``cd azkaban-solo-server/build/install/azkaban-solo-server; bin/start-solo.sh`` // you should not see any output.
* Navigate to ``http://127.0.0.1:8081/``. You should see Azkaban UI. The username is "azkaban" and password is "azkaban".
* Write down your backend IP address. (on linux, you can use ``ifconfig`` to check your IP)

*If you are using SLURM or AWS backends. Step 2 is not necessary.* 
## Step 3 (Prepare the job script)
Take a look at the file: ``examples/pytorch/wlm/start_wlm_azkaban.sh``

## Step 4 (Prepare the tuning config)
Take a look at this file: ``examples/pytorch/wlm/wlm_azkaban.py``

## Step 5 (Start tuning)
From the client machine run:

``python tune.py --config=examples/pytorch/wlm/wlm_azkaban.py --verbose 3``

Note that, in this example, client and backend machines can be one.

## Results
By default, the results will be saved in the file ``results.csv``. They will be ordered with the top results first.
Also, it will be updated on the fly, as results come in.
If a job failed for whatever reason, it will be still logged in ``results.csv`` with ``inf`` as a result.

**For AWS example, checkout [this tutorial](Quick_start_aws.md)**.