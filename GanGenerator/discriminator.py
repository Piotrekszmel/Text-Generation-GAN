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
        raise ValueError(f"Linear is expecting 2D arguments: %s".format(shape))
    if not shape[1]:
        raise ValueError(f"Linear expects shape[1] of arguments: %s".format(shape))
    input_size = shape[1]

    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)
    
    return tf.matmul(input_, tf.transpose(matrix)) + bias_term



