# How to add new Search Algorithm to Milano

One of the goals of Milano is to enable Auto-ML and Meta-learning researchers
to test their algorithms.

We provide an abstract class `Milano.search_algorithms.base.SearchAlgorithm` which integrates
meta-learning algorithms into Milano infrastructure.

### To add your own meta-learning algorithm to Milano you need:
* In the implementation of your algorithm, inherit from `Milano.search_algorithms.base.SearchAlgorithm`
* Put your algorithm under `Milano.search_algorithms`
* In your tuning config, specify your algorithm.

Here is an example on how to integrate GPSearch algorithm from [spearmint](https://github.com/JasperSnoek/spearmint/) into Milano.

**WARNING:** The steps below will bring GPL_v3 dependencies into the code. 
  
* The code under `Milano/search_algorithms/gp/spearmint` is from [spearmint's original repo](https://github.com/JasperSnoek/spearmint/). It was edited to work with Python 3.x
* The `Milano/search_algorithms/gp/gp_search.py` file contains a `GPSearch` class which inherits from `Milano.search_algorithms.base.SearchAlgorithm` and calls spearmint's code from it's functions.
It supports the following parameters:
  * **chooser**: a spearmint chooser class. For the full list of available
  choosers, see [Milano/search_algorithms/gp/spearmint](https://gitlab-dl.nvidia.com/dl-algo/Milano/tree/master/Milano/search_algorithms/gp/spearmint).
  The original [spearmint](https://github.com/JasperSnoek/spearmint/) repository
  has some advices on when to use which chooser.
  * **chooser_params**: a dictionary with chooser parameters. All of the choosers
  accept `covar` parameter which specifies which covariance function to use for the
  underlying Gaussian process. See [Milano/search_algorithms/gp/spearmint/gp.py](https://gitlab-dl.nvidia.com/dl-algo/Milano/tree/master/Milano/search_algorithms/gp/spearmint/gp.py)
  for the list of all available covariance functions. Additionally all choosers
  accept `mcmc_iters` parameter (number of MCMC iterations to run) and `noiseless`
  parameter (True of False, whether your function is evaluated exactly or we are
  only given a noisy estimate of the true function being optimized).
  * **num_init_jobs**: number of jobs to generate initially. In almost all cases
  you should set it equal to the number of workers used in backend.
  * **num_jobs_to_launch_each_time**: number of jobs to launch after each function
  evaluation is available. In most cases you should simply set it to $1$.
  * **grid_size**: grid size to sample the parameter space from. Increasing 
  grid size should generally increase your model capacity and give better results,
  but it will also make each next point generation more time consuming.
  * **smooth_inf_to**: number to use instead of `np.inf` which is returned when
  constraints are violated. Set it to `np.inf` for GPConstrainedEIChooser, since
  it can natively handle infinities and set it to some big number for other
  choosers so that they will not "want" to sample the constraint-violated points
  again.

### Run simple example
* Examine `Milano/search_algorithms/gp` folder. It already contains the wrapper `gp_search.py` and 
all the necessary code from the he original [spearmint](https://github.com/JasperSnoek/spearmint/) repository under
`Milano/search_algorithms/gp/spearmint`.
* Modify [language model tuning example](Quick_start.md) so that it uses this algorithm:
    * Add this line on top of the config file: `from Milano.search_algorithms import GPSearch`
    * Change `search_algorithm = RandomSearch` to `search_algorithm = GPSearch`
    * Add `"num_init_jobs": X,` where `X` is number of workers in your Azkaban config or equal to `"num_workers":` value in your `backend_params` if you are using AWS or SLURM.