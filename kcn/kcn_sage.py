import numpy as np
import sklearn
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Dropout, Dense
from keras.models import Model

from spektral.layers import GraphSageConv
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.utils import Batch, batch_iterator

from prepare_data import load_kriging_data

flags = tf.app.flags
FLAGS = flags.FLAGS

SW_KEY = 'dense_1_sample_weights:0'  # Keras automatically creates a placeholder for sample weights, which must be fed

def get_sub_graph(coords, features, y_a, y_f, nbs, mask):
    n = mask.shape[0]

    X__, A__, mask__, y__ = [], [], [], []

    for i in range(n):
        if mask[i] == False:
            continue

        inbs = nbs[i]

        # for y__
        y__.append(y_a[inbs].astype(np.float32))

        # for X__
        y_nbs = y_f[inbs]
        input_y = y_nbs[:, 0:1].copy()
        input_y[0] = 0.0
        indicator = np.zeros(input_y.shape)
        indicator[0] = 1.0
        gfeat = np.concatenate([features[inbs], input_y, indicator], axis=1)
        X__.append(gfeat.astype(np.float32))

        # for A__
        gcoords = coords[inbs]
        gadj = sklearn.metrics.pairwise.rbf_kernel(gcoords, gamma=1.0 / 2.0)
        A__.append(gadj.astype(np.float32))

        # for mask__
        gmask = np.zeros(y_nbs.shape[0], dtype=bool)
        gmask[0] = True
        mask__.append(gmask.astype(np.float32))

    return X__, A__, mask__, y__

def run_kcn_sage():

    def evaluate(A_list, X_list, y_list, mask_list, ops, batch_size):
        batches_ = batch_iterator([A_list, X_list, y_list, mask_list], batch_size=batch_size)
        output_ = []
        y_ = []

        for b_ in batches_:
            batch_ = Batch(b_[0], b_[1])
            X__, A__, _ = batch_.get('XAI')
            y__ = np.vstack(b_[2])
            mask__= np.concatenate(b_[3], axis=0)
            feed_dict_ = {X_in: X__,
                          A_in: sp_matrix_to_sp_tensor_value(A__),
                          mask_in: mask__,
                          target: y__,
                          SW_KEY: np.ones((1,))}

            outs_ = sess.run(ops, feed_dict=feed_dict_)

            output_.append(outs_[1][mask__.astype(np.bool)])
            y_.append(y__[mask__.astype(np.bool)])

        output_ = np.concatenate(output_, axis=0)
        y_ = np.concatenate(y_, axis=0)

        mse = (output_[:, 0] - y_[:, 0]) ** 2

        return mse.mean(), np.std(mse) / np.sqrt(mse.shape[0])


    ################################################################################
    # LOAD DATA
    ################################################################################

    coords, features, y, y_train_val, nbs, Ntrain, train_mask, val_mask, test_mask = load_kriging_data(FLAGS.dataset, FLAGS.n_neighbors)

    y_f_train = y * train_mask[:, np.newaxis].astype(np.float)
    X_train, A_train, mask_train, y_train = get_sub_graph(coords, features, y, y_f_train, nbs, train_mask)
    X_val, A_val, mask_val, y_val = get_sub_graph(coords, features, y, y_f_train, nbs, val_mask)
    X_test, A_test, mask_test, y_test = get_sub_graph(coords, features, y, y_train_val, nbs, test_mask)

    # Parameters
    F = X_train[0].shape[-1]  # Dimension of node features
    n_out = y_train[0].shape[-1]  # Dimension of the target

    ################################################################################
    # BUILD MODEL
    ################################################################################
    X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, F), name='X_in'))
    A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True, name='A_in')
    mask_in = Input(tensor=tf.placeholder(tf.float32), shape=(None, ), name='mask_in')
    target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_out), name='target'))

    # Block 1
    gc1 = GraphSageConv(FLAGS.hidden1,
                        aggregate_method='max',
                        activation=keras.activations.relu,
                        use_bias=True)([X_in, A_in])
    gc1 = Dropout(FLAGS.dropout)(gc1)

    if FLAGS.hidden2 != -1:
        # Block 2
        gc2 = GraphSageConv(FLAGS.hidden2,
                            aggregate_method='max',
                            activation=keras.activations.relu,
                            use_bias=True)([gc1, A_in])
        gc2 = Dropout(FLAGS.dropout)(gc2)
    else:
        gc2 = gc1

    # Output block
    output = Dense(n_out, activation=FLAGS.last_activation, use_bias=True)(gc2)

    # Build model
    model = Model([X_in, A_in], output)
    model.compile(optimizer='adam', loss='mse', loss_weights=[mask_in], target_tensors=[target])

    # Training setup
    sess = K.get_session()
    loss = model.total_loss
    opt = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    train_step = opt.minimize(loss)

    # Initialize all variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    ################################################################################
    # FIT MODEL
    ################################################################################
    # Run training loop
    current_batch = 0
    model_loss = 0
    model_acc = 0
    best_val_loss = np.inf
    patience = FLAGS.es_patience
    batches_in_epoch = np.ceil(len(y_train) / FLAGS.batch_size)

    print('Fitting model')
    batches = batch_iterator([A_train, X_train, y_train, mask_train], batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    for b in batches:
        batch = Batch(b[0], b[1])
        X_, A_, _ = batch.get('XAI')
        y_ = np.vstack(b[2])
        mask_ = np.concatenate(b[3], axis=0)

        tr_feed_dict = {X_in: X_,
                        A_in: sp_matrix_to_sp_tensor_value(A_),
                        mask_in: mask_,
                        target: y_,
                        SW_KEY: np.ones((1,))}
        outs = sess.run([train_step, loss], feed_dict=tr_feed_dict)

        model_loss += np.sum(outs[1] * mask_)

        current_batch += 1
        if current_batch % batches_in_epoch == 0:
            model_loss /= np.sum(train_mask)

            # Compute validation loss and accuracy
            val_loss, val_loss_std = evaluate(A_val, X_val, y_val, mask_val, [loss, output], batch_size=FLAGS.batch_size)


            ep = int(current_batch / batches_in_epoch)

            print('Ep: {:d} - Train loss: {:.5f} - Val mse: {:.5f}'.format(ep, model_loss, val_loss))

            # Check if loss improved for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = FLAGS.es_patience
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping (best val_loss: {})'.format(best_val_loss))
                    break
            model_loss = 0

    ################################################################################
    # EVALUATE MODEL
    ################################################################################
    # Test model
    test_loss, test_loss_std = evaluate(A_test, X_test, y_test, mask_test, [loss, output], batch_size=FLAGS.batch_size)
    print('Test mse: {:.5f}'.format(test_loss))

