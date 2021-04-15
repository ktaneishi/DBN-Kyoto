# Deep Belief Network for Predicting Compound-Protein Interactions

## Introduction

DBN-Kyoto is an in-silico drug discovery workload using a Deep Belief Network (DBN). This workload, so-called _virtual screening_, is used to predict whether a chemical compound interacts with a given protein sequence or not. In this case, the deep learning method resolved the serious bottleneck of exponential increase of the calculation time and memory consumption which we encountered when we applied a SVM algorithm.

This implementation was used for the optimization of [Theano by Intel](https://github.com/intel/theano), and now this optimization code was happily merged in the original [Theano](https://github.com/theano/theano).

DBN-Koyot was published in [CGBVS-DNN: Prediction of Compound-protein Interactions Based on Deep Learning, MolInf. 2016.](http://onlinelibrary.wiley.com/doi/10.1002/minf.201600045/abstract)

## Environments

- Theano-1.0.5

## Dataset

- cpi.npz - Example dataset of compound protein interactions.

## Usage

```
$ python main.py
```
