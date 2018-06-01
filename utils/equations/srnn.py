import tensorflow as tf 

def get_srnn_weights(name, inputs_dim, units, batch_size, hidden_prob):
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['w_x'] = tf.get_variable('w_x', [inputs_dim, units])
        weights['u_h'] = tf.get_variable('u_h', [units, units])
        weights['bias'] = tf.get_variable('bias', [units])
        dummy_dp = tf.ones([batch_size, units])
        weights['dropout'] = [tf.nn.dropout(dummy_dp, hidden_prob) for _ in range(1)]
    return weights

def srnn(prev, x, weights, backward=False): # prev = c+h
    #prev_c, prev_h = tf.unstack(prev, 2, 0) # [batch_size, units]
    prev_h = prev
    if backward:
        non_paddings = tf.reshape(x[1], [-1, 1])## [b, 1] 
        x = x[0] ## [b, d] ## for backward path, x is a list with two elts

    h = tf.nn.tanh(tf.matmul(prev_h**weights['dropout'][0], weights['u_h'])+tf.matmul(x, weights['w_x'])+weights['bias'])

    cell_hidden = h
    #cell_hidden = tf.stack([prev_c, prev_h])
    if backward:
        cell_hidden = cell_hidden*non_paddings

    return  cell_hidden


