# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing


class CNN(nn.Module):  #define the inheritance
    def __init__(self, number_actions):
        super(CNN,self).__init__()      #activate the inheritance
        self.convolution1 = nn.Conv2d(in_channels = 1,out_channels = 32,kernel_size=5)  # in_channels => (1 if black and white), 
        self.convolution2 = nn.Conv2d(in_channels = 32,out_channels = 32,kernel_size=3) # 
        self.convolution3 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size=2) # Wreduce the kernel to detect smaller features
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features=40)
        self.fc2 = nn.Linear(in_features = 40, out_features= number_actions)
    
    def count_neurons(self, image_dim):
        #DOOM images are going to be of size 80X80
        # Create random fake image
        
        x = Variable(torch.rand(1,*image_dim))
        #propagating random image over all conv layers
        x = F.relu(F.max_pool2d(self.convolution1(x),3,2)) #3 and 2 are kernel size and stride respectively
        x = F.relu(F.max_pool2d(self.convolution2(x),3,2))
        x = F.relu(F.max_pool2d(self.convolution3(x),3,2))
        
        #flattening the image and getting number of neurons        
        return x.data.view(1,-1).size(1)
    
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.convolution1(x),3,2))
        x = F.relu(F.max_pool2d(self.convolution2(x),3,2))
        x = F.relu(F.max_pool2d(self.convolution3(x),3,2))
        
        #flattening
        x = x.view(x.size(0),-1)
        #To learn non linearity (breaking linearity with relu)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SoftmaxBody(nn.Module):
    def __init__(self, T):  #Temperature
        super(SoftmaxBody,self).__init__()      #activate the inheritance
        self.T = T
    
    def forward(self,outputs):
        probabilities = F.softmax(outputs*self.T)
        actions = probabilities.multinomial()
        return actions

class AI:
    def __init__(self,brain,body):
        self.brain = brain
        self.body = body
        
    
    def __call__(self,inputs):
        inputs =Variable(torch.from_numpy(np.array(inputs,dtype = np.float32)))
        outputs=self.brain(inputs)
        actions = self.body(outputs)    
        return actions.data.numpy()
    


# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)
number_actions = doom_env.action_space.n
    
  
# Building an AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)  
    
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        
    state = series[0].state
    target = output[0].data
    target[series[0].action] = cumul_reward
    inputs.append(state)
    targets.append(target)
     
    return torch.from_numpy(np.array(inputs,dtype = np.float32)), torch.stack(targets)

# Moving average on 100 steps

class MA:
    def __init__(self,size):
        self.list_of_rewards = []
        self.size = size
        
    def add(self, rewards):
        if isinstance(rewards,list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        
        while (len(self.list_of_rewards) > self.size) :
            del self.list_of_rewards[0]
            
    
    def average(self):
        return np.mean(self.list_of_rewards)   
    
    
ma = MA(100)