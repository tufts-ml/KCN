from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'loss_type'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss_type = kwargs.get('loss_type')

        self.loss = 0
        self.train_error = 0.0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.debug = [] 

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            

            self.activations.append(hidden)


        #self.debug.append(self.layers[0].vars['att_wts_0'])

        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[-1]
        self.placeholders = placeholders

        #self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)

        self.build()

    def _loss(self):

        # Cross entropy error
        if self.loss_type == 'cross_ent':
            self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                   self.placeholders['labels_mask'])

        elif self.loss_type == 'squared_error':
            print('Fitting data using squared error')

            self.loss += masked_squared_error(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'], self)

        elif self.loss_type == 'zpoisson_nll':
            self.loss += masked_zpoisson_nll(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'], self)

        elif self.loss_type == 'pseudo_gaussian_error': # TODO: implement this function in metrics.py
            self.loss += masked_pseudo_gaussian_error(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask']) # use self.debug to show debugging variables
 
        else:
            raise Exception('No such loss: %s' % self.loss_type)
    
        self.train_error = self.loss

        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)




    def _accuracy(self):


        if self.loss_type == 'cross_ent':
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

        elif self.loss_type == 'squared_error':
            self.accuracy = masked_squared_error(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'], self)

        elif self.loss_type == 'zpoisson_nll':
            self.accuracy = masked_zpoisson_nll(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'], self)

        elif self.loss_type == 'pseudo_gaussian_error': # TODO: implement this function in metrics.py
            self.accuracy = masked_pseudo_gaussian_error(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask']) # use self.debug to show debugging variables

        else:
            raise Exception('No such loss: %s' % self.loss_type)

    def _build(self):


        if FLAGS.last_activation == 'exp':
            act_func = tf.exp
        elif FLAGS.last_activation == 'identity':
            act_func = lambda x: x
        elif FLAGS.last_activation == 'relu':
            act_func = tf.nn.relu
        else:
            raise Exception('No such activation function %s' % FLAGS.last_act)



        hidden_size = [FLAGS.hidden1]
        
        if not FLAGS.gcn_kriging:
            hidden_size.append(self.output_dim)
        else:
            hidden_size.append(FLAGS.hidden2)
            hidden_size.append(self.output_dim)
            
        
        if FLAGS.use_attention:
            GraphLayer = GATLayer
        else:
            GraphLayer = GraphConvolution


        ilayer = 0
        self.layers.append(GraphLayer(input_dim=self.input_dim,
                                            output_dim=hidden_size[ilayer],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=FLAGS.sparse_input,
                                            logging=self.logging))

        
        ilayer += 1
        self.layers.append(GraphLayer(input_dim=hidden_size[ilayer - 1],
                                            output_dim=hidden_size[ilayer],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=FLAGS.sparse_input,
                                            logging=self.logging))


        
           
        if FLAGS.gcn_kriging:
            ilayer += 1
            self.layers.append(Dense(input_dim=hidden_size[ilayer - 1],
                               output_dim=hidden_size[ilayer],
                               placeholders=self.placeholders,
                               act=act_func,
                               dropout=True,
                               sparse_inputs=FLAGS.sparse_input,
                               logging=self.logging))

 

    def predict(self):

        if self.loss_type == 'cross_ent':
            pred = tf.nn.softmax(self.outputs)
        elif self.loss_type == 'squared_error':
            pred = self.outputs
        elif self.loss_type == 'zpoisson_nll':
            pred = self.outputs
        else:
            raise Exception('No such loss: %s' % self.loss_type)

        return pred
