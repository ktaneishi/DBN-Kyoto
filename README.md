# DBN-Kyoto (Deep Belief Network for Drug Discovery)

## Introdusion

DBN-Kyoto is an in-silico drug discovery workload using Deep Belief Network (DBN). This workload, so called _virtual screening_, is used to predict whether a chemical compound interacts with a given protein sequence or not. In this case, deep learning method resolve the serious bottleneck of exponential increasing of the calculation time and memory consumption which we encountered when we applied SVM algorithm.

This implementation were used for the optimization of [Theano by Intel](https://github.com/intel/theano), and now this optimization were happily merged in original [Theano](https://github.com/theano/theano).

The academic results using DBN-Kyoto were reported in our paper [CGBVS-DNN: Prediction of Compound-protein Interactions Based on Deep Learning, MolInf. 2016.](http://onlinelibrary.wiley.com/doi/10.1002/minf.201600045/abstract).

## Dependency

- Theano-1.0 or later.

## Files

- dbn.py
    Definition of Deep Belief Network.

- main.py
    Main script.

- benchmark.py
    Benchmark script.

- cpi.npz  
    Sample data of compound protein interactions (shrinked because of github file size limitation).  

## Usage

```
$ python benchmark.py
```
