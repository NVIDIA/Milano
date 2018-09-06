# Milano 
(*This is a research project, not an official NVIDIA product.*)

<div align="center">
  <img src="iconMilano.png" alt="Milano" width="250px">
  <br>
</div>

Milano (Machine learning autotuner and network optimizer) is a tool for enabling machine learning researchers and practitioners to perform massive hyperparameters 
and architecture searches.

You can use it to:
* Tune your model on a cloud backend of your choice
* Benchmark Auto-ML algorithms (see [how to add new search algorithm](docs/how-to-add-new-search-algorithm.md))

Your script can use any framework of your choice, for example, TensorFlow, PyTorch, Microsoft Cognitive Toolkit etc. or no framework at all.
Milano only requires minimal changes to what your script accepts via command line and what it returns to stdout. 

**Currently supported backends:**
* Azkaban - on a single multi-GPU machine or server with Azkaban installed
* AWS - Amazon cloud using GPU instances
* SLURM - any cluster which is running SLURM

### Prerequisites

* Linux
* Python 3
* Ensure you have Python version 3.5 or later with packages listed in the `requirements.txt` file.
* Backend with NVIDIA GPU

## How to Get Started
1. Install all dependencies with the following command
   pip install -r requirements.txt.
2. Follow this [mini-tutorial](docs/Quick_start.md).

## Documentation
[https://nvidia.github.io/Milano](https://nvidia.github.io/Milano)


