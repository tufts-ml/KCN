import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    avg_loss = tf.reduce_mean(tf.boolean_mask(loss, mask))

    return avg_loss

def masked_squared_error(preds, labels, mask, model):
    """squared error with masking."""


    labels = tf.slice(tf.boolean_mask(labels, mask), [0, 0], [-1, 1]) 
    preds = tf.slice(tf.boolean_mask(preds, mask), [0, 0], [-1, 1])

    loss = tf.squared_difference(preds, labels)
    avg_loss = tf.reduce_mean(loss)


    return avg_loss

def masked_zpoisson_nll(preds, labels, mask, model):
    """squared error with masking."""


    labels = tf.slice(tf.boolean_mask(labels, mask), [0, 0], [-1, 1]) 

    preds = tf.boolean_mask(preds, mask)

    logits = tf.slice(preds, [0, 0], [-1, 1])   
    log_lamb = tf.tanh(tf.slice(preds, [0, 1], [-1, 1])) * 5.0 # the range of lambda is exp(-5) to exp(5) 

    poisson_loss = tf.nn.log_poisson_loss(labels, log_lamb)
    bin1_loss = tf.math.softplus(- logits)

    positive_loss = poisson_loss + bin1_loss

    bin0_loss = tf.math.softplus(logits)
    loss_concat = tf.concat([positive_loss, bin0_loss], axis=1)
    zero_loss = tf.reduce_logsumexp(loss_concat, axis=1, keep_dims=True)
    
    loss = tf.where(labels > 0.01, positive_loss, zero_loss)

    assert_op = tf.Assert(tf.less_equal(tf.reduce_sum(loss), 1e9), [labels[0], preds[0], loss[0], zero_loss[0], log_lamb[0], logits[0], poisson_loss[0]])

    with tf.control_dependencies([assert_op]):
        avg_loss = tf.reduce_mean(loss)

    # model.debug = 

    return avg_loss




def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)

    acc = tf.reduce_mean(tf.boolean_mask(accuracy_all, mask))

    return acc
