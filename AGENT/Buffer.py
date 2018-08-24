import numpy as np
class Buffer(object):
    def __init__(self,config):
        self.surob=np.zeros([config.buffer_size,config.screen_height,config.screen_width,config.screen_channals],dtype=np.float32)
        self.reward=np.zeros([config.buffer_size,1],dtype=np.float64)
        self.flag=np.zeros([config.buffer_size,1],dtype=np.int8)
        self.action=np.zeros([config.buffer_size,1],dtype=np.int32)
        self.surob_next=np.zeros([config.buffer_size,config.screen_height,config.screen_width,config.screen_channals],dtype=np.float32)
        self.size=config.buffer_size
        self.i=0

    def clean_buffer(self):
        buffer_size,screen_width,screen_height,screen_channals=np.shape(self.surob)
        for i in range(buffer_size):
            self.surob[i]=np.zeros([screen_width,screen_height,screen_channals],dtype=np.float32)
            self.reward[i]=np.zeros([1],dtype=np.float64)
            self.flag[i]=np.zeros([1],dtype=np.int8)
            self.action[i]=np.zeros([1],dtype=np.int32)
            self.surob_next[i]=np.zeros([screen_width,screen_height,screen_channals],dtype=np.float32)


    def add_buffer(self,surob,reward,flag,action,surob_next):
        i = self.i
        self.surob[i]=surob
        self.reward[i]=reward
        self.flag[i]=flag
        self.action[i]=action
        self.surob_next[i]=surob_next
        self.i = (self.i + 1) % self.size

