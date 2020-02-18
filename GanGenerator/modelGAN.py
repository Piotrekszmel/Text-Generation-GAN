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
            self.g_worker_recurrent_unit = self.create_Worker_recurrent_unit(self.worker_params)
            self.g_worker_output_unit = self.create_Worker_output_unit(self.worker_params)
            self.W_workerOut_change = tf.Variable(tf.random_normal([self.vocab_size, self.goal_size], stddev=0.1))

            self.g_change = tf.Variable(tf.random_normal([self.goal_out_size, self.goal_size], stddev=0.1))
            self.worker_params.extend([self.W_workerOut_change,self.g_change])

            self.h0_worker = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_worker = tf.stack([self.h0_worker, self.h0_worker])

        with tf.variable_scope('Manager'):
            self.g_manager_recurrent_unit = self.create_Manager_recurrent_unit(self.manager_params)
            self.g_manager_output_unit = self.create_Manager_output_unit(self.manager_params)
            self.h0_manager = tf.zeros([self.batch_size, self.hidden_dim])
            self.h0_manager = tf.stack([self.h0_manager, self.h0_manager])

            self.goal_init = tf.get_variable("goal_init",initializer=tf.truncated_normal([self.batch_size,self.goal_out_size], stddev=0.1))
            self.manager_params.extend([self.goal_init])
        
        self.padding_array = tf.constant(-1, shape=[self.batch_size, self.sequence_length], dtype=tf.int32)

        with tf.name_scope("roll_out"):
            self.gen_for_reward = self.rollout(self.x, self.given_num)

        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x),perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        gen_o = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                 dynamic_size=False, infer_shape=True)
        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32,size=1,dynamic_size=True, infer_shape=True,clear_after_read = False)

        goal = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                 dynamic_size=False, infer_shape=True,clear_after_read = False)

        feature_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length+1,
                                                     dynamic_size=False, infer_shape=True, clear_after_read=False)
        real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_real_goal_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        gen_o_worker_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        def _g_recurrence(i, x_t,h_tm1,h_tm1_manager, gen_o, gen_x,goal,last_goal,real_goal,step_size,gen_real_goal_array,gen_o_worker_array):
            ## padding sentence by -1
            cur_sen = tf.cond(i > 0,lambda:tf.split(tf.concat([tf.transpose(gen_x.stack(), perm=[1, 0]),self.padding_array],1),[self.sequence_length,i],1)[0],lambda :self.padding_array)
            with tf.variable_scope(self.scope):
                feature = self.FeatureExtractor_unit(cur_sen,self.drop_out)
            h_t_Worker = self.g_worker_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t_Worker = self.g_worker_output_unit(h_t_Worker)  # batch x vocab , logits not prob
            o_t_Worker = tf.reshape(o_t_Worker,[self.batch_size,self.vocab_size,self.goal_size])

            h_t_manager = self.g_manager_recurrent_unit(feature,h_tm1_manager)
            sub_goal = self.g_manager_output_unit(h_t_manager)
            sub_goal = tf.nn.l2_normalize(sub_goal, 1)
            goal = goal.write(i,sub_goal)

            real_sub_goal = tf.add(last_goal,sub_goal)

            w_g = tf.matmul(real_goal,self.g_change)   #batch x goal_size
            w_g = tf.nn.l2_normalize(w_g, 1)
            gen_real_goal_array = gen_real_goal_array.write(i,real_goal)

            w_g = tf.expand_dims(w_g,2)  #batch x goal_size x 1

            gen_o_worker_array = gen_o_worker_array.write(i,o_t_Worker)

            x_logits = tf.matmul(o_t_Worker,w_g)
            x_logits = tf.squeeze(x_logits)

            log_prob = tf.log(tf.nn.softmax(
                tf.cond(i > 1, lambda: tf.cond(self.train > 0, lambda: self.tem, lambda: 1.5), lambda: 1.5) * x_logits))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            with tf.control_dependencies([cur_sen]):
                gen_x = gen_x.write(i, next_token)  # indices, batch_size
            gen_o = gen_o.write(i, tf.reduce_sum(tf.multiply(tf.one_hot(next_token, self.vocab_size, 1.0, 0.0),
                                                             tf.nn.softmax(x_logits)), 1))  # [batch_size] , prob
            return i+1,x_tp1,h_t_Worker,h_t_manager,gen_o,gen_x,goal,\
                   tf.cond(((i+1)%step_size)>0,lambda:real_sub_goal,lambda :tf.constant(0.0,shape=[self.batch_size,self.goal_out_size]))\
                    ,tf.cond(((i+1)%step_size)>0,lambda :real_goal,lambda :real_sub_goal),step_size,gen_real_goal_array,gen_o_worker_array

        _, _, _,_, self.gen_o, self.gen_x,_,_,_,_,self.gen_real_goal_array,self.gen_o_worker_array= control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4,_5,_6,_7,_8,_9,_10,_11: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token),self.h0_worker,self.h0_manager,
                        gen_o, gen_x,goal,tf.zeros([self.batch_size,self.goal_out_size]),self.goal_init,step_size,gen_real_goal_array,gen_o_worker_array),parallel_iterations=1)

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size

        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

        self.gen_real_goal_array = self.gen_real_goal_array.stack()  # seq_length x batch_size x goal

        self.gen_real_goal_array = tf.transpose(self.gen_real_goal_array, perm=[1, 0,2])  # batch_size x seq_length x goal

        self.gen_o_worker_array = self.gen_o_worker_array.stack()  # seq_length x batch_size* vocab*goal

        self.gen_o_worker_array = tf.transpose(self.gen_o_worker_array, perm=[1, 0,2,3])  # batch_size x seq_length * vocab*goal

        sub_feature = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length/self.step_size,
                                                       dynamic_size=False, infer_shape=True, clear_after_read=False)

        all_sub_features = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)
        all_sub_goals = tensor_array_ops.TensorArray(dtype=tf.float32, size=self.sequence_length,
                                                   dynamic_size=False, infer_shape=True, clear_after_read=False)

        # supervised pretraining for generator
        g_predictions = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length,
            dynamic_size=False, infer_shape=True)
        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)
                                            