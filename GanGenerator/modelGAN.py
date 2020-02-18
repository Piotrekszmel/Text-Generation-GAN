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

        with tf.variable_scope("place_holder"):
            self.x = tf.placeholder(tf.int32, shape=[self.batch_size,self.sequence_length])
            self.reward = tf.placeholder(tf.float32, shape=[self.batch_size,self.sequence_length / self.step_size])
            self.given_num = tf.placeholder(tf.int32)
            self.drop_out = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.train = tf.placeholder(tf.int32, None, name="train")
        
        with tf.variable_scope('Worker'):
            self.g_embeddings = tf.Variable(tf.random_normal([self.vocab_size, self.emb_dim], stddev=0.1))
            self.worker_params.append(self.g_embeddings)
            self.g_worker_recurrent_unit = self.create_Worker_recurrent_unit(self.worker_params)  # maps h_tm1 to h_t for generator
            self.g_worker_output_unit = self.create_Worker_output_unit(self.worker_params)  # maps h_t to o_t (output token logits)
            self.W_workerOut_change = tf.Variable(tf.random_normal([self.vocab_size, self.goal_size], stddev=0.1))

            self.g_change = tf.Variable(tf.random_normal([self.goal_out_size, self.goal_size], stddev=0.1))
            self.worker_params.extend([self.W_workerOut_change,self.g_change])

            self.h0_worker = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_worker = tf.stack([self.h0_worker, self.h0_worker])

        with tf.variable_scope('Manager'):
            self.g_manager_recurrent_unit = self.create_Manager_recurrent_unit(self.manager_params)  # maps h_tm1 to h_t for generator
            self.g_manager_output_unit = self.create_Manager_output_unit(self.manager_params)  # maps h_t to o_t (output token logits)
            self.h0_manager = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_manager = tf.stack([self.h0_manager, self.h0_manager])

            self.goal_init = tf.get_variable("goal_init",initializer=tf.truncated_normal([self.batch_size,self.goal_out_size], stddev=0.1))
            self.manager_params.extend([self.goal_init])
        