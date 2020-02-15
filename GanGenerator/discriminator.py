from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow as tf 
import numpy as np 


def linear(input_, output_size, scope=None):
    """
    Args:
    input_: a tensor or a list of 2D [batch, n] Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    
    Returns:
    A 2D Tensor with shape [batch, output_size]
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {}".format(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {}".format(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term

# The highway layer is borrowed from https://github.com/mkroutikov/tf-lstm-char-cnn
def highway(input_,. size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope="Highway"):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope="highway_lin_{}".format(idx)))

            t = tf.sigmoid(linear(input_, size, scope="highway_gate_{}".format(idx)) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output


