import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import random
import math

class QNet(nn.Module):
    def __init__(self,hidden_layer,action_size,station_size,isgpu):
        super().__init__()
        input_space=station_size
        out_space=action_size
        if isgpu:
            self.layer0=nn.Linear(input_space,hidden_layer).cuda()
            self.layer1=nn.Linear(hidden_layer,out_space).cuda()
        else:
            self.layer0=nn.Linear(input_space,hidden_layer)
            self.layer1=nn.Linear(hidden_layer,hidden_layer)
            self.layer2=nn.Linear(hidden_layer,out_space)
    
    def forward(self, input):

        X=F.tanh(self.layer0(input))
        X=F.tanh(self.layer1(X))
        action_score=self.layer2(X)
        # out_select=F.softmax(action_score,dim=1)
        return action_score


class Solution():
    def __init__(self,hidden_layer,gamma,
    eps_start,eps_end,eps_decay,
    LR,batch_size,cache_len,update_C,
    action_size,station_size,isgpu):

        self.sample_cache=[]
        self.cache_len=cache_len
        self.action_size=action_size
        self.station_size=station_size
        self.gamma=gamma
        self.batch_size=batch_size
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay=eps_decay
        self.steps_done=0
        self.update_C=update_C
        
        self.qnet=QNet(hidden_layer,action_size,station_size,isgpu)
        self.qnet_t=QNet(hidden_layer,action_size,station_size,isgpu)

        self.opt=optim.Adam(self.qnet.parameters(),lr=LR)
    
    def push_cache(self,sample_o):
        s,a,r,s_next=sample_o
        s_t=torch.FloatTensor([s])
        a_t=torch.LongTensor([a])
        r_t=torch.FloatTensor([r])
        s_next_t=torch.FloatTensor([s_next])
        sample=[s_t,a_t,r_t,s_next_t]

        if len(self.sample_cache)<self.cache_len:
            self.sample_cache.append(sample)
        else:
            self.sample_cache.pop(0)
            self.sample_cache.append(sample)
    
    def pop_samples(self):
        return random.sample(self.sample_cache, self.batch_size)
    
    def select_action(self,state,isdone):
        sample=random.random()
        eps_threshold=self.eps_end+(self.eps_start-self.eps_end)*math.exp(
            -1.0*self.steps_done/self.eps_decay
        )
        self.steps_done+=1
        if sample>eps_threshold or isdone:
            a=Variable(torch.FloatTensor([state]))
            a=self.qnet(a).data.max(1)[1].item()
            return a
        else:
            return random.randrange(2)


    
    def train_step(self):

        samples=self.pop_samples()
        s,a,r,s_next=zip(*samples)
        s=Variable(torch.cat(s))
        a=Variable(torch.cat(a))
        a=a.reshape([self.batch_size,1])
        r=Variable(torch.cat(r))
        s_next=Variable(torch.cat(s_next))
        current_q=self.qnet.forward(s)
        current_q=current_q.gather(1,a)
        max_next_q=self.qnet_t.forward(s_next).detach().max(1)[0]
        y_t=(r+self.gamma*max_next_q).reshape([self.batch_size,1])
        
        loss=F.smooth_l1_loss(current_q,y_t)
        loss_data=loss.item()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.steps_done%self.update_C==0:
            self.qnet_t.load_state_dict(self.qnet.state_dict())

        return loss_data
