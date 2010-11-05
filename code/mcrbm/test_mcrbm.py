import sys
from pylearn.algorithms.mcRBM import *
import pylearn.datasets.cifar10
import pylearn.dataset_ops.tinyimages

import pylearn.dataset_ops.cifar10
from theano import tensor
from pylearn.shared.layers.logreg import LogisticRegression

def l2(X):
    return numpy.sqrt((X**2).sum())

def _default_rbm_alloc(n_I, n_K=256, n_J=100):
    return mcRBM.alloc(n_I, n_K, n_J)

def _default_trainer_alloc(rbm, train_batch, batchsize, l1_penalty, l1_penalty_start):
    return mcRBMTrainer.alloc(rbm, train_batch, batchsize, l1_penalty=l1_penalty,
            l1_penalty_start=l1_penalty_start,persistent_chains=persistent_chains)

def test_reproduce_ranzato_hinton_2010(dataset='MAR', 
        n_train_iters=5000,
        rbm_alloc=_default_rbm_alloc, 
        trainer_alloc=_default_trainer_alloc,
        lr_per_example=.075,
        l1_penalty=1e-3,
        l1_penalty_start=1000,
        persistent_chains=True,
        ):

    batchsize = 128
    ## specific to MAR dataset ##
    n_vis=105
    n_patches=10240
    epoch_size=n_patches

    tile = pylearn.dataset_ops.image_patches.save_filters_of_ranzato_hinton_2010

    batch_idx = TT.iscalar()
    batch_range =batch_idx * batchsize + np.arange(batchsize)

    train_batch = pylearn.dataset_ops.image_patches.ranzato_hinton_2010_op(batch_range)

    imgs_fn = function([batch_idx], outputs=train_batch)

    trainer = trainer_alloc(
            rbm_alloc(n_I=n_vis),
            train_batch,
            batchsize, 
            initial_lr_per_example=lr_per_example,
            l1_penalty=l1_penalty,
            l1_penalty_start=l1_penalty_start,
            persistent_chains=persistent_chains)
    rbm=trainer.rbm

    if persistent_chains:
        grads = trainer.contrastive_grads()
        learn_fn = function([batch_idx], 
                outputs=[grads[0].norm(2), grads[0].norm(2), grads[1].norm(2)],
                updates=trainer.cd_updates())
    else:
        learn_fn = function([batch_idx], outputs=[], updates=trainer.cd_updates())

    if persistent_chains:
        smplr = trainer.sampler
    else:
        smplr = trainer._last_cd1_sampler

    if dataset == 'cifar10patches8x8':
        cPickle.dump(
                pylearn.dataset_ops.cifar10.random_cifar_patches_pca(
                    n_vis, None, 'float32', n_patches, R, C,),
                open('test_mcRBM.pca.pkl','w'))

    print "Learning..."
    last_epoch = -1
    for jj in xrange(n_train_iters):
        epoch = jj*batchsize / epoch_size

        print_jj = epoch != last_epoch
        last_epoch = epoch

        if print_jj:
            tile(imgs_fn(jj), "imgs_%06i.png"%jj)
            if persistent_chains:
                tile(smplr.positions.value, "sample_%06i.png"%jj)
            tile(rbm.U.value.T, "U_%06i.png"%jj)
            tile(rbm.W.value.T, "W_%06i.png"%jj)

            print 'saving samples', jj, 'epoch', jj/(epoch_size/batchsize)

            print 'l2(U)', l2(rbm.U.value),
            print 'l2(W)', l2(rbm.W.value),
            print 'l1_penalty', 
            try:
                print trainer.effective_l1_penalty.value
            except:
                print trainer.effective_l1_penalty

            print 'U min max', rbm.U.value.min(), rbm.U.value.max(),
            print 'W min max', rbm.W.value.min(), rbm.W.value.max(),
            print 'a min max', rbm.a.value.min(), rbm.a.value.max(),
            print 'b min max', rbm.b.value.min(), rbm.b.value.max(),
            print 'c min max', rbm.c.value.min(), rbm.c.value.max()

            if persistent_chains:
                print 'parts min', smplr.positions.value.min(), 
                print 'max',smplr.positions.value.max(),
            print 'HMC step', smplr.stepsize.value,
            print 'arate', smplr.avg_acceptance_rate.value


        l2_of_Ugrad = learn_fn(jj)

        if persistent_chains and print_jj:
            print 'l2(U_grad)', float(l2_of_Ugrad[0]),
            print 'l2(U_inc)', float(l2_of_Ugrad[1]),
            print 'l2(W_inc)', float(l2_of_Ugrad[2]),
            #print 'FE+', float(l2_of_Ugrad[2]),
            #print 'FE+[0]', float(l2_of_Ugrad[3]),
            #print 'FE+[1]', float(l2_of_Ugrad[4]),
            #print 'FE+[2]', float(l2_of_Ugrad[5]),
            #print 'FE+[3]', float(l2_of_Ugrad[6])

        if jj % 2000 == 0:
            print ''
            print 'Saving rbm...'
            cPickle.dump(rbm, open('mcRBM.rbm.%06i.pkl'%jj, 'w'), -1)
            if persistent_chains:
                print 'Saving sampler...'
                cPickle.dump(smplr, open('mcRBM.smplr.%06i.pkl'%jj, 'w'), -1)


    return rbm, smplr
