import sys
import time
import numpy as np
import sklearn
import tensorflow as tf
import scipy

from utils import *
from models import GCN

from prepare_data import load_kriging_data

flags = tf.app.flags
FLAGS = flags.FLAGS

def run_kcn(use_attention):
    # setting
    flags.DEFINE_boolean('use_attention', use_attention, 'Use attention in gcn.')
    flags.DEFINE_boolean('gcn_kriging', True, 'Use gcn kriging.')
    flags.DEFINE_boolean('sparse_input', False, 'Use sparse matrices.')

    # Load data

    n_neighbors = FLAGS.n_neighbors
    batch_size = FLAGS.batch_size
    length_scale = FLAGS.length_scale

    coords, features, y, y_train_val, nbs, Ntrain, _, _, _ = load_kriging_data(FLAGS.dataset, n_neighbors)

    print('feature dimensions: ', features.shape)

    input_dim = features.shape[1] + 2

    # Some preprocessing
    num_supports = 1

    model_func = GCN

    # Define placeholders
    placeholders = {
        'support': [tf.placeholder(tf.float32, shape=[None, n_neighbors + 1, n_neighbors + 1]) for _ in
                    range(num_supports)],
        'features': tf.placeholder(tf.float32, shape=[None, n_neighbors + 1, input_dim]),
        'labels': tf.placeholder(tf.float32, shape=(None, n_neighbors + 1, y.shape[1])),
        'labels_mask': tf.placeholder(tf.int32, shape=[None, n_neighbors + 1]),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=input_dim, logging=True, loss_type=FLAGS.loss_type)

    print('good up to here')
    # Initialize session
    sess = tf.Session()


    def prepare_input(batch_ind, nbs, labels, coords):

        y_nbs_list = []
        gfeat_list = []
        support_list = []
        gmask_list = []

        for ind in batch_ind:
            inbs = nbs[ind]

            y_nbs = labels[inbs]
            y_nbs_list.append(y_nbs)

            input_y = y_nbs[:, 0:1].copy()
            input_y[0] = 0.0
            indicator = np.zeros(input_y.shape)
            indicator[0] = 1.0
            gfeat = np.concatenate([features[inbs], input_y, indicator], axis=1)
            gfeat_list.append(gfeat)

            gcoords = coords[inbs]
            gadj = sklearn.metrics.pairwise.rbf_kernel(gcoords, gamma=1.0 / (2.0 * length_scale * length_scale))
            support = preprocess_adj(gadj)
            support_list.append(support)

            gmask = np.zeros(y_nbs.shape[0], dtype=bool)
            gmask[0] = True
            gmask_list.append(gmask)

        y_nbs_ = np.stack(y_nbs_list, axis=0)
        gfeat_ = np.stack(gfeat_list, axis=0)
        support_ = [np.stack(support_list, axis=0)]
        gmask_ = np.stack(gmask_list, axis=0)

        return y_nbs_, gfeat_, support_, gmask_


    # Define model evaluation function
    def evaluate(features, nbs, y_train_val, ins_range, placeholders):
        t_test = time.time()

        loss = 0.0
        error = []

        for i in ins_range:
            y_nbs, gfeat, support, gmask = prepare_input([i], nbs, y_train_val, coords)

            feed_dict_val = construct_feed_dict(gfeat, support, y_nbs, gmask, placeholders)

            # accuracy is from previous GCN implementation. It actually records errors
            outs_val = sess.run([model.loss, model.accuracy, model.debug], feed_dict=feed_dict_val)
            loss = loss + outs_val[0]

            error.append(outs_val[1])

            # debug

        # if (len(ins_range) > 2000):
        #     print('saving matrices')
        #     #np.savetxt('weight-att.csv', outs_val[2][0], delimiter=',')
        #     np.savetxt('no-att-error.csv', error, delimiter=',')

        avg_loss = loss / len(ins_range)

        avg_error = np.mean(error)
        var_error = np.var(error)

        return avg_error, var_error, (time.time() - t_test)


    saver = tf.train.Saver()

    # Init variables
    sess.run(tf.global_variables_initializer())

    valid_error_list = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()

        train_loss = 0.0
        train_error = 0.0

        num_train = Ntrain - 500
        for i in range(0, num_train, batch_size):
            batch_ind = range(i, min(i + batch_size, num_train))

            # Construct feed dictionary
            y_nbs, gfeat, support, gmask = prepare_input(batch_ind, nbs, y_train_val, coords)

            feed_dict = construct_feed_dict(gfeat, support, y_nbs, gmask, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.train_error, model.debug], feed_dict=feed_dict)

            train_loss = train_loss + outs[1] * len(batch_ind)
            train_error = train_error + outs[2] * len(batch_ind)

            # if i in [0, 1, 2, 3, 4, 5, 8, 10]:
            #    print('training debug')
            #    print(outs[3])

        avg_train_loss = train_loss / num_train
        avg_train_error = train_error / num_train

        # Validation
        valid_ind = range(num_train, Ntrain)
        valid_error, valid_variance, duration = evaluate(features, nbs, y_train_val, valid_ind, placeholders)
        valid_error_list.append(valid_error)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_train_loss),
              "val_mse=", "{:.5f}".format(valid_error), "time=", "{:.5f}".format(time.time() - t))

        if epoch > FLAGS.es_patience and np.mean(valid_error_list[-3:]) > np.mean(
                valid_error_list[-(FLAGS.es_patience + 3):-3]):
            print("Early stopping...")
            break

    last_valid_error = np.mean(valid_error_list[-4:])

    print("Optimization Finished!")

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    # save_path = saver.save(sess, "kcn-no-att.ckpt")
    # print("Model saved in path: %s" % save_path)

    # Testing
    num_test = y.shape[0] - Ntrain
    test_ind = range(Ntrain, Ntrain + num_test)
    test_error, test_variance, test_duration = evaluate(features, nbs, y, test_ind, placeholders)
    print("Test set results:",
          "mse=", "{:.5f}".format(test_error), "time=", "{:.5f}".format(test_duration))


