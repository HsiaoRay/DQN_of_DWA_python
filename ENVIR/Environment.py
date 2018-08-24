# import gym
import math
import numpy as np
import ctypes
import cv2

class resData(ctypes.Structure):
    _fields_ = [("upState",ctypes.POINTER(ctypes.c_uint8)),
                ("upNextState",ctypes.POINTER(ctypes.c_uint8)),
                ("dReward",ctypes.c_double),
                ("bflag",ctypes.c_bool)]

def add_robotpos(surob,angle):
  pic = surob.copy()
  centerx = surob.shape[0]/2
  centery = surob.shape[1]/2
  fp = np.zeros((2),dtype='uint8')
  fp[0] = round(5 * math.cos(angle) + centerx)
  fp[1] = round(5 * math.sin(angle) + centery)
  cv2.line(pic,(centerx,centery),(fp[0],fp[1]),127)
  return pic

class Environment(object):
  def __init__(self, config):
    self.config=config
    self.dims = (config.screen_width, config.screen_height)
    self._screen = None
    self.reward = 0.0
    self.preang = 0.0
    self.sumang = 0.0
    ll = ctypes.cdll.LoadLibrary
    self.lib=ll("/home/moro/Desktop/Dev_LowAI/RoboticSys/EwayOS/Function/Test/libEnvForPy.so")
    self.lib.Step.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double]
    self.lib.Step.restype = resData
    self.lib.GetCurPosInfo.restype = ctypes.POINTER(ctypes.c_double)
    self.is_fail(is_newgame=1,curang=0.0)

  def get_nextstate(self, action):
    w1 = action%self.config.action_size1
    action = int(action / self.config.action_size1)
    w2 = action%self.config.action_size2
    action = int(action/self.config.action_size2)
    w3 = action%self.config.action_size3
    w1 = (self.config.w1_max-self.config.w1_min)/self.config.action_size1 * w1 + self.config.w1_min
    w2 = (self.config.w2_max-self.config.w2_min)/self.config.action_size1 * w2 + self.config.w2_min
    w3 = (self.config.w3_max-self.config.w3_min)/self.config.action_size1 * w3 + self.config.w3_min
    w=[1.0,w1,w2,w3]
    #w=[1.0,1.0,1.0,0.1]
    #print("%f,%f,%f,%f"%(w[0],w[1],w[2],w[3]))
    #C++ get next position
    # surob=np.random.rand(self.config.screen_height,self.config.screen_width,self.config.screen_channals)
    # surob_next=np.random.rand(self.config.screen_height,self.config.screen_width,self.config.screen_channals)
    # State = np.zeros(shape=(self.config.screen_height,self.config.screen_width,self.config.screen_channals),dtype = 'uint8')

    cur_reward = 0.01
    flag = 0

    pos = self.lib.GetCurPosInfo()
    predis =(pos[5]-pos[3])**2+(pos[4]-pos[2])**2

    rd = self.lib.Step(w[0],w[1],w[2],w[3])
    surob = self.ucharp2ndArray(rd.upState)
    surob_next = self.ucharp2ndArray(rd.upNextState)
    surob[:,:,0] = add_robotpos(surob[:,:,0],pos[6])
    surob_next[:,:,0] = add_robotpos(surob_next[:,:,0],pos[6])

    surob = surob / 255.0
    surob.astype(np.float32)
    surob_next=surob_next / 255.0
    surob_next.astype(np.float32)

    pos = self.lib.GetCurPosInfo()
    curdis = (pos[5]-pos[3])**2+(pos[4]-pos[2])**2

    cirflag = self.is_fail(is_newgame = False,curang=pos[6])

    if (rd.bflag == 1 and rd.dReward == 5):
      cur_reward = 10.0
      self.is_fail(is_newgame=True,curang=0.0)
      flag = 1
    else:
      #cur_reward =  math.exp(-curdis/20)
      if (curdis > predis):
        cur_reward = -1.0
      if (cirflag > 0):
        cur_reward = -0.5
        if (cirflag == 2):
          # self.start_newgame()
          self.is_fail(is_newgame=True,curang=0.0)
          flag = -1
          print('Circle Fail!')

    return surob,cur_reward,flag,surob_next

  def get_pos_info(self):
    p = self.lib.GetCurPosInfo()
    # Pos = np.zeros(shape=(1,6),dtype='float64')
    Pos=[]
    for i in range(7):
      Pos.append(p[i])
    return Pos

  def start_newgame(self):
    self.lib.Restart()
    print('Restart a new game!')

  def ucharp2ndArray(self,up):
    State = np.zeros(shape=(self.config.screen_height,self.config.screen_width,self.config.screen_channals),dtype = 'uint8')
    pos = 0
    for ch in range(2):
        for i in range(100):
            for j in range(100):
                State[i][j][ch]=up[pos]
                pos=pos+1
    return State

  @property
  def action_size(self):
    #return self.env.action_space.n
    return self.config.action_size1*self.config.action_size2*self.config.action_size3



  def is_fail(self,is_newgame,curang):
    flag = 0
    if(is_newgame == True):
      self.sumang = 0.0
      self.preang = curang
    else:
      self.sumang = self.sumang+curang-self.preang
      if(self.preang>3 and curang<-3):
        self.sumang = self.sumang+2.0*3.1415926
      if(self.preang<-3 and curang>3):
        self.sumang = self.sumang-2.0*3.1415926
      self.preang = curang
      if (self.sumang > 6.3 or self.sumang < -6.3):
        flag = 1
      if (self.sumang > 12 or self.sumang < -12):
        flag = 2
    # print(self.sumang)
    return flag



