import sys
import time
from multiprocessing import Process, Queue

import yaml
import numpy as np
import zmq

# Jinlong added: 2015-10-27
import logging
# set up logging to file - see previous section for more details

from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M:%S',
                    filename='./temp/alexnet_time.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

# Now, define a couple of other loggers which might represent areas in your
# application:
logger = logging.getLogger('AlexNet.timming')

sys.path.append('./lib')
from tools import (save_weights, load_weights,
                   save_momentums, load_momentums)
from train_funcs import (unpack_configs, adjust_learning_rate,
                         get_val_error_loss, get_rand3d, train_model_wrap,
                         proc_configs)


def train_net(config):
    # UNPACK CONFIGS
    (flag_para_load, train_filenames, val_filenames,
     train_labels, val_labels, img_mean) = unpack_configs(config)
    if flag_para_load:
        #  zmq set up
        sock = zmq.Context().socket(zmq.PAIR)
        sock.connect('tcp://localhost:{0}'.format(config['sock_data']))

        load_send_queue = config['queue_t2l']
        load_recv_queue = config['queue_l2t']
    else:
        load_send_queue = None
        load_recv_queue = None

    import theano
    theano.config.on_unused_input = 'warn'

    if config['flag_top_5']:    # Jinlong added 2015-10-30
        flag_top5 = True
    else:
        flag_top5 = False 

    from layers import DropoutLayer
    from alex_net import AlexNet, compile_models

    ## BUILD NETWORK ##
    model = AlexNet(config)
    layers = model.layers
    batch_size = model.batch_size

    ## COMPILE FUNCTIONS ##
    (train_model, validate_model, train_error, learning_rate,
        shared_x, shared_y, rand_arr, vels) = compile_models(model, config, flag_top_5=flag_top5)


    ######################### TRAIN MODEL ################################

    print '... training'

    #print shared_x.type
    if flag_para_load:
        # [2015-10-21]
        # Jinlong added to support parallel load for CPU version
        sock.send_pyobj((shared_x))
        load_send_queue.put(img_mean)

    n_train_batches = len(train_filenames)
    minibatch_range = range(n_train_batches)



    # Start Training Loop
    epoch = 0
    step_idx = 0
    val_record = []
    while epoch < config['n_epochs']:
        epoch = epoch + 1

        if config['shuffle']:
	    print ('shuffle')
            np.random.shuffle(minibatch_range)

        if config['resume_train'] and epoch == 1:
	    print ('config')
            load_epoch = config['load_epoch']
            load_weights(layers, config['weights_dir'], load_epoch)
            #print layers[0].params[1].get_value()
            #sys.exit(0)
            epoch = load_epoch + 1
            lr_to_load = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
            #val_record = list(
            #    np.load(config['weights_dir'] + 'val_record.npy'))
            learning_rate.set_value(lr_to_load)
            load_momentums(vels, config['weights_dir'], load_epoch)
            #load_momentums(vels, config['weights_dir'], epoch)

        if flag_para_load:
	    print ('flag_para_load')
            # send the initial message to load data, before each epoch
            load_send_queue.put(str(train_filenames[minibatch_range[0]]))
            load_send_queue.put(get_rand3d())

            # clear the sync before 1st calc
            load_send_queue.put('calc_finished')

        count = 0
        #for minibatch_index in minibatch_range[:1]:
        for minibatch_index in minibatch_range:

            num_iter = (epoch - 1) * n_train_batches + count
            count = count + 1
            #print str(datetime.today()) + "  @iter " + str(count)
            if count == 1:
                s = time.time()
            if count == 20:
                e = time.time()
                print "time per 20 iter:", (e - s)
            cost_ij = train_model_wrap(train_model, shared_x,
                                       shared_y, rand_arr, img_mean,
                                       count, minibatch_index,
                                       minibatch_range, batch_size,
                                       train_filenames, train_labels,
                                       flag_para_load,
                                       config['batch_crop_mirror'],
                                       send_queue=load_send_queue,
                                       recv_queue=load_recv_queue)

            #print shared_x.get_value(borrow=True)

            if num_iter % config['print_freq'] == 0:
                #print 'training @ iter = ', num_iter
                #print 'training cost:', cost_ij
		logger.info("training @ iter = %i" % (num_iter)) 
		logger.info("training cost: %lf" % (cost_ij)) 
                if config['print_train_error']:
                    logger.info('training error rate: %lf' % train_error())
                    #print 'training error rate:', train_error()

            if flag_para_load and (count < len(minibatch_range)):
                load_send_queue.put('calc_finished')

        ############### Test on Validation Set ##################

        #"""
        DropoutLayer.SetDropoutOff()

        # result_list = [ this_validation_error, this_validation_error_top5, this_validation_loss ]
        # or
        # result_list = [ this_validation_error, this_validation_loss ]
        result_list = get_val_error_loss(
        #this_validation_error, this_validation_loss = get_val_error_loss(
            rand_arr, shared_x, shared_y,
            val_filenames, val_labels,
            flag_para_load, img_mean,
            batch_size, validate_model,
            send_queue=load_send_queue, 
            recv_queue=load_recv_queue,
            flag_top_5=flag_top5)


        logger.info(('epoch %i: validation loss %f ' %
              (epoch, result_list[-1])))
        #print('epoch %i: validation loss %f ' %
        #      (epoch, this_validation_loss))
        if flag_top5:
            logger.info(('epoch %i: validation error (top 1) %f %%, (top5) %f %%' %
                (epoch,  result_list[0] * 100., result_list[1] * 100.)))
        else:
            logger.info(('epoch %i: validation error %f %%' %
                (epoch, result_list[0] * 100.)))
        #print('epoch %i: validation error %f %%' %
        #      (epoch, this_validation_error * 100.))
        val_record.append(result_list)
        #val_record.append([this_validation_error, this_validation_loss])
        np.save(config['weights_dir'] + 'val_record.npy', val_record)

        DropoutLayer.SetDropoutOn()
        ############################################

        # Adapt Learning Rate
        step_idx = adjust_learning_rate(config, epoch, step_idx,
                                        val_record, learning_rate)

        # Save weights
        if epoch % config['snapshot_freq'] == 0:
            save_weights(layers, config['weights_dir'], epoch)
            np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate.get_value())
            save_momentums(vels, config['weights_dir'], epoch)
        #"""

    print('Optimization complete.')


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    with open('spec.yaml', 'r') as f:
        config = dict(config.items() + yaml.load(f).items())
        
    config = proc_configs(config)

    if config['para_load']:
        from proc_load import fun_load
        config['queue_l2t'] = Queue(1)
        config['queue_t2l'] = Queue(1)
        train_proc = Process(target=train_net, args=(config,))
	print 'config : ',config
        load_proc = Process(
            target=fun_load, args=(config, config['sock_data']))
        train_proc.start()
        load_proc.start()
        train_proc.join()
        load_proc.join()

    else:
        """
        train_proc = Process(target=train_net, args=(config,))
        train_proc.start()
        train_proc.join()
        """
        train_net(config) # For theano profiling
