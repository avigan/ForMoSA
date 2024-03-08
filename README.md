<p align="left"><img src="docs/source/ForMoSA.png" alt="ForMoSA" width="250"/></p>


***
Installation
===

We strongly recommend using a ``conda`` environment ([learn more here](https://conda.io/docs/user-guide/tasks/manage-environments.html)).

To install and use our package in macOS with an M1 chip, proceed with the following sequence of commands:

1/ To run parallelization without warnings, make sure you are building your environment under an OSX-ARM64 architecture ([learn more here](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple)).

    $ CONDA_SUBDIR=osx-arm64 conda create -n env_formosa python=3.11 numpy -c conda-forge

    $ conda activate env_formosa 
    
    $ conda config --env --set subdir osx-arm64

2/ Install the following packages in your environment with pip install: 
    
    $ numpy
    $ matplotlib
    $ corner
    $ astropy
    $ scipy
    $ configobj
    $ extinction
    $ nestle
    $ PyAstronomy
    $ spectres
    $ pyyaml
    $ importlib-metadata==4.13.0
    $ xarray==2023.10.1


3/ To solve a standard error, run the following line in your environment:

    $ conda install xarray dask netCDF4 bottleneck

4/ If you want to use Pymultinest to run your inversion, follow the installation instructions from [PyMultinest](https://johannesbuchner.github.io/PyMultiNest/install.html), detailed below. 

If you don't need PyMultinest, you can directly go to point 5/.

First. Clone PyMultinest from GitHub and install it.

    $ git clone https://github.com/JohannesBuchner/PyMultiNest/
    $ cd PyMultiNest
    $ python setup.py install

Second. Make sure your system has a C++ and a Fortran interpreter. If you need to install brew, follow [these instructions](https://brew.sh/).

    $ brew install cmake
    $ brew install gcc
    $ brew install open-mpi

Third. In your ForMoSA environment, install mpi4pi as:
    
    $ pip install mpi4py

Fourth. Install MultiNest by cloning the GitHub repository and building it. Make sure you empty the build folder if you run this step more than once.
    
    $ git clone https://github.com/JohannesBuchner/MultiNest
    $ cd MultiNest/build
    $ cmake ..
    $ make

Finally, copy the files that were generated by building MultiNest onto your conda (may be miniconda) environment by doing:

    $ cp -v ~/YOUR_PATH/MultiNest/lib/* /YOUR_PATH/opt/anaconda3/envs/env_formosa/lib/
	

5/ The last instalation step is to actually get ForMoSA. ForMoSA can be installed throgh pip, however, to work on the final released version, please clone the main branch from our GitHub repository. You can do it by moving to the desired directory on your terminal and writing:

    $ git clone https://github.com/exoAtmospheres/ForMoSA.git


***
Running the code
===

Follow the instructions in ForMoSA/DEMO

***
Issues?
===

If you encounter any other problem, please create an issue on GitHub ([https://github.com/exoAtmospheres/ForMoSA/issues](https://github.com/exoAtmospheres/ForMoSA/issues/new)).

***

[![Documentation Status](https://readthedocs.org/projects/formosa/badge/?version=latest)](https://formosa.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/formosa.svg)](https://badge.fury.io/py/formosa)
[![PyPI downloads](https://img.shields.io/pypi/dm/formosa.svg)](https://pypistats.org/packages/formosa)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![A rectangular badge, half black half purple containing the text made at Code Astro](https://img.shields.io/badge/Made%20at-Code/Astro-blueviolet.svg)](https://semaphorep.github.io/codeastro/)
