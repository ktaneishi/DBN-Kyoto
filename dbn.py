import numpy as np
import timeit
import sys
import os

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM
from DBN import DBN

def load_data(dataset, nfold=5):
    print('... loading data')
    data = np.load(dataset)['data']

    np.random.seed(123)
    data = np.random.permutation(data)

    train_set = data[:-int(data.shape[0] / nfold)]
    test_set = data[-int(data.shape[0] / nfold):]

    def shared_dataset(data_xy, borrow=True):
        data_x = data_xy[:,:-1]
        data_y = data_xy[:,-1]
        shared_x = theano.shared(
                np.asarray(data_x, dtype=theano.config.floatX),
                borrow=borrow)
        shared_y = theano.shared(
                np.asarray(data_y, dtype=theano.config.floatX),
                borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

def build_finetune_functions(self, datasets, batch_size, learning_rate):
    (train_set_x, train_set_y) = datasets[0]
    (test_set_x, test_set_y) = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches //= batch_size

    index = T.lscalar('index')  # index to a [mini]batch

    # compute the gradients with respect to the model parameters
    gparams = T.grad(self.finetune_cost, self.params)

    # compute list of fine-tuning updates
    updates = []
    for param, gparam in zip(self.params, gparams):
        updates.append((param, param - gparam * learning_rate))

    train_fn = theano.function(
        inputs=[index],
        outputs=self.finetune_cost,
        updates=updates,
        givens={
            self.x: train_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            self.y: train_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }
    )

    test_score_i = theano.function(
        [index],
        self.errors,
        givens={
            self.x: test_set_x[
                index * batch_size: (index + 1) * batch_size
            ],
            self.y: test_set_y[
                index * batch_size: (index + 1) * batch_size
            ]
        }
    )

    # Create a function that scans the entire test set
    def test_score():
        return [test_score_i(i) for i in range(n_test_batches)]

    return train_fn, test_score

def versions():
    versions = (
            ('theano', theano.version.version),
            ('numpy', np.version.version),
            ('python', sys.version))
    return versions

def show_version():
    for version in versions():
        print(' '.join(version))

if __name__ == '__main__':
    show_version()
