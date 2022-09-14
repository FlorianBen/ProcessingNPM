# ProcessingNPM

- [ProcessingNPM](#processingnpm)
  - [Summary](#summary)
  - [Directories](#directories)
  - [Dependencies](#dependencies)
  - [Installation](#installation)


## Summary

This repository contains Python scripts for the processing of NPM data. It consists of a Python library and several use cases.

## Directories

The repository is organized in the following manner:

| Directory | Description                            |
| --------- | -------------------------------------- |
| npm       | Library directory                      |
| docs      | Documentation building directory       |
| use_case  | Several use cases that use the library |

## Dependencies

The scripts require several Python packages in order to run correctly. Be sure that the following Python packages are installed:
* numpy
* scipy
* pandas
* matplotlib
* h5py
* opencv-python
* pywavelets
* uproot
* pyFFTW

One can use the `pip install -r requirements.txt` command to install the necessary dependencies. If you want to build the code source documentation, then `Sphinx` should be installed as well. 

## Installation

As the library is in a constant development, one can use `pip install -e .` in this top directory. Then, one may be able to run the use cases. 