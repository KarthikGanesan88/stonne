# STONNE: A Simulation Tool for Neural Networks Engines

Custom fork of the STONNE project for my own use. The original repo downloads PyTorch 1.7.0 to pass network parameters to STONNE. Since we are already upto Pytorch v1.10.0 (as of Nov '21), I wanted to decouple STONNE from a specific PyTorch version. 

This version keeps the PyTorch changes contained to a few files so it will work with any version of PyTorch (assuming they dont change something as basic as Conv2D and Linear layers at any point!)

Please make sure to checkout the original repo for the full details and be sure to cite the authors if you use STONNE in your work! 

## Main changes

1. Removed `pytorch-frontend` folder and moved just the 2 modified files `SimulatedConv.py` and `SimulatedLinear.py` into a new folder `pytorch_stonne`. 
2. Modified the paths in all files to work with current directory structure. 
3. Wrote a small function to automatically replace regular `conv2d` and `linear` layers with simulated ones with matching sizes. If you want to run the network in PyTorch to get accuracy, you can quickly copy the network, replace layers and also run it via stonne.

## Usage

The usage of STONNE remains the same. I only changed the pytorch frontend. For an example, check out `/pytorch_stonne/test.py` which creates a model and calls a function to automatically convert it to a simulated copy to generate the files needed for stonne. 
