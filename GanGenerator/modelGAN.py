import tensorflow as tf
import  numpy as np
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class GeneratorGan:
    def __init__(self, sequence_length, num_classes, vocab_size,
            emb_dim, dis_emb_dim,filter_sizes, num_filters,batch_size,hidden_dim, start_token,goal_out_size,
                 goal_size,step_size,D_model,LSTMlayer_num=1, l2_reg_lambda=0.0,learning_rate=0.001):
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dis_emb_dim = dis_emb_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.start_token = tf.constant([start_token] * self.batch_size, dtype=tf.int32)
        self.LSTMlayer_num = LSTMlayer_num
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.num_filters_total = sum(self.num_filters)
        self.grad_clip = 5.0
        self.goal_out_size = goal_out_size
        self.goal_size = goal_size
        self.step_size = step_size
        self.D_model = D_model
        self.FeatureExtractor_unit = self.D_model.FeatureExtractor_unit

        self.scope = self.D_model.feature_scope
        self.worker_params = []
        self.manager_params = []

        self.epis = 0.65
        self.tem = 0.8