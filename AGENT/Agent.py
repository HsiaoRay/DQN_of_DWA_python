# from __future__ import print_function
import os
import time
import random
import numpy as np
# from tqdm import tqdm
import tensorflow as tf

from .Nodedef import conv2d,linear,clipped_error
from .Buffer import Buffer

class Agent(object):
    def __init__(self, config, environment, sess):
        self.config = config
        self.sess = sess
        self.weight_dir = 'weights'
        self.env = environment
        self.buffer=Buffer(config)

        #building the net
        self.build_dqn()
        if (config.new_model == False):
            self.load_model()


        self.buffer.clean_buffer()

        with open(config.log_path,'w+') as f:
            f.write('Start Trainning!\n')
        with open(config.log_pathb,'w+') as f:
            f.write('Start Trainning!\n')


    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        #initializer = tf.contrib.layers.xavier_initializer()
        initializer = tf.truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu

        # training network
        with tf.variable_scope('prediction'):
          if self.config.cnn_format == 'NHWC':
            self.s_t = tf.placeholder('float32',
                [None, self.config.screen_height, self.config.screen_width, self.config.screen_channals], name='s_t')
          else:
            self.s_t = tf.placeholder('float32',
                [None, self.config.screen_channals, self.config.screen_height, self.config.screen_width], name='s_t')

          self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
              32, [8, 8], [4, 4], initializer, activation_fn, self.config.cnn_format, name='l1')
          self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
              64, [4, 4], [2, 2], initializer, activation_fn, self.config.cnn_format, name='l2')
          self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
              64, [3, 3], [1, 1], initializer, activation_fn, self.config.cnn_format, name='l3')

          shape = self.l3.get_shape().as_list()
          self.l3_flat = tf.reshape(self.l3, [-1, reduce(lambda x, y: x * y, shape[1:])])
          self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=activation_fn, name='l4')
          self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_size, name='q')

          #self.q_action = tf.argmax(self.q, dimension=1)

          q_summary = []
          avg_q = tf.reduce_mean(self.q, 0)
          for idx in xrange(self.env.action_size):
            q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
          self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        '''
        # target network
        with tf.variable_scope('target'):
          if self.config.cnn_format == 'NHWC':
            self.target_s_t = tf.placeholder('float32',
                [None, self.config.screen_height, self.config.screen_width, self.config.history_length], name='target_s_t')
          else:
            self.target_s_t = tf.placeholder('float32',
                [None, self.config.history_length, self.config.screen_height, self.config.screen_width], name='target_s_t')

          self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
              32, [8, 8], [4, 4], initializer, activation_fn, self.config.cnn_format, name='target_l1')
          self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.target_l1,
              64, [4, 4], [2, 2], initializer, activation_fn, self.config.cnn_format, name='target_l2')
          self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.target_l2,
              64, [3, 3], [1, 1], initializer, activation_fn, self.config.cnn_format, name='target_l3')

        shape = self.target_l3.get_shape().as_list()
        self.target_l3_flat = tf.reshape(self.target_l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                linear(self.target_l3_flat, 512, activation_fn=activation_fn, name='target_l4')
        self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                linear(self.target_l4, self.env.action_size, name='target_q')

        self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

        with tf.variable_scope('pred_to_target'):
          self.t_w_input = {}
          self.t_w_assign_op = {}

          for name in self.w.keys():
            self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
            self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])
        '''

    # optimizer
        with tf.variable_scope('optimizer'):
          self.target_q_t = tf.placeholder('float32', [None,self.env.action_size], name='target_q_t')
          #self.action = tf.placeholder('int64', [None], name='action')

          #action_one_hot = tf.one_hot(self.action, self.config.action_size, 1.0, 0.0, name='action_one_hot')
          #q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')

          self.delta = self.target_q_t - self.q

          self.global_step = tf.Variable(0, trainable=False)

          self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')
          self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
          self.learning_rate_op = 0.01
          # self.optim=tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)
          self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        tf.global_variables_initializer().run()


    def get_action(self,surob,num):
        ep = self.config.ep_start*self.config.anneal_rate**num
        ep = max(ep,self.config.ep_end)
        if self.config.cnn_format == "NCHW":
            surob = np.transpose(surob,(2,0,1))

        out=self.q.eval({self.s_t: [surob]})[0]
        if random.random() < ep:
            action = random.randrange(self.env.action_size)
        else:
            action = np.argmax(out)
        return action, max(out)

    def learn(self,surob,reward,flag,surob_next,action,num,step):
        # add current info to the buffer
        self.buffer.add_buffer(surob,reward,flag,action,surob_next)
        loss = 0
        # if buffer is full, rolling states/rewaards from buffer and training them
        ## in order to imitate a target-net by a singer net, we set a large number for training frequency
        ## and when it's time to train the net, training it for several times to utilize buffer adequately
        # The target using methods above both is to make the simulation and training asynchronously
        ##imagining another simple trainning method: get the state/reward. train the net, and using the new
        ##net directly, then each time we change the output of the net (for ex, larger), the net will generate a
        ##more latge output, which causes the net instable
        if (num>self.config.buffer_size and num%self.config.train_frequency==0):
            for tt in range(self.config.train_num):
                # generating a random rank, and rolling states/rewards from buffer according to the rank
                rank=np.random.permutation(self.config.buffer_size)
                cur_surob=np.zeros([self.config.batch_size,self.config.screen_height,self.config.screen_width,self.config.screen_channals],dtype=np.float32)
                cur_reward=np.zeros([self.config.batch_size,1],dtype=np.float32)
                cur_flag=np.zeros([self.config.batch_size,1],dtype=np.int8)
                cur_action=np.zeros([self.config.batch_size,1],dtype=np.int8)
                cur_surob_next=np.zeros([self.config.batch_size,self.config.screen_height,self.config.screen_width,self.config.screen_channals],dtype=np.float32)
                for i in range(self.config.batch_size):
                    pos=rank[i]
                    cur_surob[i]=self.buffer.surob[pos]
                    cur_reward[i]=self.buffer.reward[pos]
                    cur_flag[i]=self.buffer.flag[pos]
                    cur_action[i]=self.buffer.action[pos]
                    cur_surob_next[i]=self.buffer.surob_next[pos]

                if self.config.cnn_format == "NCHW":
                    cur_surob = np.transpose(cur_surob,(0,3,1,2))
                    cur_surob_next = np.transpose(cur_surob_next,(0,3,1,2))

                # getting the next q by current net
                out_next = self.q.eval({self.s_t: cur_surob_next})
                max_nextq=np.max(out_next,1)
                out=self.q.eval({self.s_t: cur_surob})
                # updating the out
                for i in range(self.config.batch_size):
                    ac=cur_action[i]
                    if (1==cur_flag[i]):
                        out[i][ac] = cur_reward[i]
                    else:
                        out[i][ac] = cur_reward[i] + self.config.gamma * max_nextq[i]
                    print(out[i][ac]),
                    print(' '),
                print(' ')
                 # updating the net
                _, loss = self.sess.run([self.optim,self.loss], feed_dict={self.s_t: cur_surob,self.target_q_t: out,self.learning_rate_step:step})
        return loss

    def save_model(self,num):
        saver=tf.train.Saver()
        str = './Model/model%d'%(num)
        if os.path.exists(str):
            for i in os.listdir(str):
                path_file = os.path.join(str,i)
                os.remove(path_file)
        else:
            os.makedirs(str)
        str = str+'/TrainModel'
        saver.save(self.sess,str)
        str = './Model/newmodel'
        if os.path.exists(str):
            for i in os.listdir(str):
                path_file = os.path.join(str,i)
                os.remove(path_file)
        else:
            os.makedirs(str)
        str = str+'/TrainModel'
        saver.save(self.sess,str)
        print('model%d is saved'%(num))

    def load_model(self):
        str = './Model/newmodel/TrainModel.meta'
        # with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(str)
        new_saver.restore(self.sess,tf.train.latest_checkpoint('./Model/newmodel/'))
        print('load model successfully')


'''
        self.load_model()
        self.update_target_q_network()
'''
