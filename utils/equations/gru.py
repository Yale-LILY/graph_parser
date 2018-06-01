import tensorflow as tf 

def get_gru_weights(name, inputs_dim, units, batch_size, hidden_prob):
    weights = {}
    with tf.variable_scope(name) as scope:
        weights['w_x_z'] = tf.get_variable('w_x_z', [inputs_dim, units])
        weights['w_x_r'] = tf.get_variable('w_x_r', [inputs_dim, units])
        weights['w_x_h'] = tf.get_variable('w_x_h', [inputs_dim, units] )
        weights['u_h_z'] = tf.get_variable('u_h_z', [units, units])
        weights['u_h_r'] = tf.get_variable('u_h_r', [units, units])
        weights['u_h_h'] = tf.get_variable('u_h_h', [units, units])
        weights['bias_z'] = tf.get_variable('bias_z', [units])
        weights['bias_r'] = tf.get_variable('bias_r', [units])
        weights['bias_h'] = tf.get_variable('bias_h', [units])
        dummy_dp = tf.ones([batch_size, units])
        weights['dropout'] = [tf.nn.dropout(dummy_dp, hidden_prob) for _ in range(3)]
    return weights

def gru(prev, x, weights, backward=False): # prev = c+h
    #prev_c, prev_h = tf.unstack(prev, 2, 0) # [batch_size, units]
    prev_h = prev
    if backward:
        non_paddings = tf.reshape(x[1], [-1, 1])## [b, 1] 
        x = x[0] ## [b, d] ## for backward path, x is a list with two elts

    z_gate = tf.nn.sigmoid(tf.matmul(prev_h*weights['dropout'][0], weights['u_h_z'])+tf.matmul(x, weights['w_x_z'])+weights['bias_z'])
    r_gate = tf.nn.sigmoid(tf.matmul(prev_h*weights['dropout'][1], weights['u_h_r'])+tf.matmul(x, weights['w_x_r'])+weights['bias_r'])
    h_gate = tf.nn.tanh(tf.matmul(prev_h*r_gate*weights['dropout'][2], weights['u_h_h'])+tf.matmul(x, weights['w_x_h'])+weights['bias_h'])
    h = z_gate*prev_h + (1.0-z_gate)*h_gate

    cell_hidden = h
    #cell_hidden = tf.stack([prev_c, prev_h])
    if backward:
        cell_hidden = cell_hidden*non_paddings

    return  cell_hidden


