import tensorflow as tf

class DQNConfig(object):
  #path
  log_path = '/home/moro/Desktop/TrainLog.txt'
  log_pathb = '/home/moro/Desktop/TrainLogBrief.txt'

  #training mode
  new_model = True
  gpu_use = True
  gpu_options = '1/1'
  gpu_fraction = 0.0
  epoch_size = 500
  num_eachepoch = 1000


  #network para
  buffer_size = 256
  batch_size = 64
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.99
  target_q_update_step = 1 * 1000
  learning_rate = 0.0005
  learning_rate_minimum = 0.00025
  learning_rate_decay = 0.96
  learning_rate_decay_step = 5 * 1000
  train_frequency = 256
  train_num = 32

  #DQN para
  screen_width = 100
  screen_height = 100
  screen_channals = 2
  action_size1 = 5
  action_size2 = 5
  action_size3 = 5
  gamma = 0.8
  display=False
  history_length = 4
  learn_start = 5. * 1000
  min_delta = -1
  max_delta = 1
  _test_step = 5 * 1000
  _save_step = _test_step * 10
  ep_end = 0.1
  ep_start = 1.0
  anneal_rate = 0.95
  w1_min = 0.0
  w1_max = 2.0
  w2_min = 0.0
  w2_max = 2.0
  w3_min = 0.0
  w3_max = 0.2

  def update(self):
    idx, num = self.gpu_options.split('/')
    idx, num = float(idx), float(num)
    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    self.gpu_fraction=fraction

    if not tf.test.is_gpu_available() and self.gpu_use:
      raise Exception("use_gpu flag is true when no GPUs are available")

    if not self.gpu_use:
      self.cnn_format = 'NHWC'


def get_config():
  config=DQNConfig()
  config.update()
  return config
