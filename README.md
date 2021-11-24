# STONNE: A Simulation Tool for Neural Networks Engines

Custom fork of the STONNE project for my own use. The original repo downloads PyTorch 1.7.0 to pass network parameters to STONNE. Since we are already upto Pytorch v1.10.0 (as of Nov '21), I wanted to decouple STONNE from a specific PyTorch version. 

This version keeps the PyTorch changes contained to a few files so it will work with any version of PyTorch (assuming they dont change something as basic as Conv2D and Linear layers at any point!)

Please make sure to checkout the original repo for the full details and be sure to cite the authors if you use STONNE in your work! 

## MAIN CHANGES

1. Removed `pytorch-frontend` folder and moved just the 2 modified files `SimulatedConv.py` and `SimulatedLinear.py` into a new folder `pytorch_stonne`. 
2. Modified the paths in all files to work with current directory structure. 
