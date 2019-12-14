import numpy as np
import tensorflow as tf
from kcn_sage import run_kcn_sage
from kcn_base import run_kcn

# fix random seed
rand_seed = 123
np.random.seed(rand_seed)
tf.set_random_seed(rand_seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'kcn', 'model to use: kcn, kcn_att, kcn_sage')
flags.DEFINE_integer('n_neighbors', 5, 'Number of neighbors')
flags.DEFINE_integer('hidden1', 5, 'Number of units in hidden layer 1')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2. -1 means not to use this layer')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability)')
flags.DEFINE_float('lr', 1e-2, 'Learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('es_patience', 10, 'Patience for early stopping')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_string('dataset', './data/data.npz', 'Data file path')
flags.DEFINE_string('last_activation', 'relu', 'Activation for the last layer')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('length_scale', 0.5, 'Length scale of the RBF kernel.')
flags.DEFINE_string('loss_type', 'squared_error', 'loss type')


if FLAGS.model == 'kcn':
    run_kcn(use_attention=False)
elif FLAGS.model == 'kcn_att':
    run_kcn(use_attention=True)
elif FLAGS.model == 'kcn_sage':
    run_kcn_sage()
else:
    print('unknown model')
