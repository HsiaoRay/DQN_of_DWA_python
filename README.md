# DQN_of_DWA_python
learning the weight of each paras in DWA(Dynamic Window Approach) by using DQN(Deep Q-Learning)
## 1、关于本项目
　使用python语言和[tensorflow](https://github.com/tensorflow/tensorflow)平台，结合C++编写的仿真环境，利用DQN(Deep Q-Learning)对DWA(Dynamic Window Approach)算法中各参数的权重进行学习。 
　代码中不包括DWA的主体部分，该部分由C++编写，通过调用该文件编译生成的.so文件得到。  
  &nbsp;
  
  
## 2、如何使用
### 　２.1 python版本
　　使用python2.x版本
### 　２.2 环境依赖
> 　　conda install opencv  
> 　　conda install numpy  
> 　　conda install tensorflow
  &nbsp;
  
  
## 3、文件与参数说明
### 　3.1 Main.py
　主训练函数，负责加载配置及调用Agent与Environment的各个模块；训练时直接运行该文件即可。 
 &nbsp;

### 　3.2 Config.py
　　配置文件，定义训练中使用到的各种参数信息。  
　　各参数信息详见注释。  
   &nbsp;

### 　3.3 AGENT/
　　对应RL（增强学习）中的Agent模块，完成学习网络的建立（新建或加载已有模型），DWA参数预测（前向传播）与更新。
#### 　　3.2.1 Agent.py
　　　网络的建立、预测与更新。    
#### 　　3.2.2 Buffer.py
　　　缓存区模块，建立和维护训练过程中得到的数据。
#### 　　3.2.3 IO.py
　　　输入输出模块，主要目的在于将训练数据写入日志文档。
#### 　　3.2.4 Nodedef.py
　　　神经网络节点的补充定义。 
   &nbsp;


###  　3.3 ENVIR/
　　对应RL（增强学习）中的Environment模块，完成对DWA算法的仿真。
  
#### 　　3.3.1 Environtment.py
　　　小车前进的状态模拟。  
　　　作为接口处理仿真环境返回的数据信息，并进行状态判断（是否转圈）与reward赋值。  
　　　关于C++部分，可联系【Email:<zhaoyuchen@ewaybot.com>】
   &nbsp;
 

## 4、其他
  目前尚未得到较好的训练结果。  
  建议修改方向：神经网络的结构、奖励的形式与大小等。  
  联系方式【Email:<zhaoxrthu@gmail.com>】
  
  
  
  
  
  

