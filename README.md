DBN-Kyoto (Deep Belief Network for Drug Discovery)
==================================================

This is a drug discovery workload using Deep Belief Network (DBN) and Deep Neural Network (DNN) model.
This algorithm is used to predict if the chemical structure of a compound can interact with the protein sequence.
Deep Learning resolve the bottleneck of exponential increase in the calculation time and memory consumption encountered in SVM algorithm.

These scripts were used for the optimization of [Theano by Intel](https://github.com/intel/theano),
and now this optimization were merged in [Theano](https://github.com/theano/theano).

The results of these models were reported in the paper _Prediction of Compound-protein Interactions Based on Deep Learning_, MolInf. 2016.
All modifications of this workload were done by the 2nd author.

Dependency
----------

- Theano (pip install theano)

Files
-----

- dbn.py  
Definition of Deep Belief Network.

- main.py  
Main script.

- benchmark.py  
Benchmark script.

- cpi.npz  
Sample data including about 250,000 of compound protein interactions.  
Download from https://my.syncplicity.com/share/vvks9oqxas1xneg/cpi

Usage
-----

    $ python benchmark.py
