from __future__ import print_function
import ctypes
import random
import cv2
import tensorflow as tf
import numpy as np

from Config import get_config
from ENVIR.Environment import Environment
from AGENT.Agent import Agent
from AGENT.IO import IO

def train():
  config = get_config()
  #gpu_options = tf.GPUself.lossOptions(per_process_gpu_memory_fraction=config.gpu_fraction)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.gpu_fraction)

  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    env = Environment(config)
    agent = Agent(config, env, sess)
    io = IO()

  #Main Train Loop
    for ep in range(config.epoch_size):
      surob=np.zeros([config.screen_height,config.screen_width,config.screen_channals],dtype=np.float32)
      reward ,fail , maxq = 0.0, 0.0, 0.0

      io.log_in(config.log_path,config.log_pathb,1,ep,config.epoch_size,None,None)
      for nu in range(config.num_eachepoch):
        # 1. predict
        action,cur_maxq = agent.get_action(surob,ep)

        # 2. forward, getting current and next states/reward
        surob,cur_reward,flag,surob_next = env.get_nextstate(action)

        # 3.learning and update
        agent.learn(surob,cur_reward,flag,surob_next,action,nu,ep)

        if (flag == 1):
          io.log_in(config.log_path,config.log_pathb,2,None,None,env.get_pos_info(),None)

        if (flag == -1):
          env.start_newgame()
          io.log_in(config.log_path,config.log_pathb,4,None,None,env.get_pos_info(),None)

        # str = '/home/moro/Desktop/pic/CurState1/%d.jpg'%nu
        # cv2.imwrite(str,255*surob[:,:,0])

        # print(cur_reward)
        # # 3.learning and update
        # agent.learn(surob,cur_reward,flag,surob_next,action,nu,ep)
        reward += cur_reward
        # fail += (flag == -1)
        maxq = maxq + cur_maxq

      reward = reward / config.num_eachepoch
      maxq = maxq/config.num_eachepoch
      str='epoch: %d/%d, sum_reward:  %f, average_maxq: %f'%(ep+1, config.epoch_size,reward,maxq)
      io.log_in(config.log_path,config.log_pathb,3,None,None,None,str)
      agent.save_model(ep)

if __name__ == '__main__':
#  tf.app.run()
   train()
