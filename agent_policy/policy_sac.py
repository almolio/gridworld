import random, math, torch 
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import deque, namedtuple
from torchrl.data import ListStorage, PrioritizedReplayBuffer, ReplayBuffer

from .net import ActorNetwork, CriticNetwork, ValueNetwork
import torch.nn.functional as F
import numpy as np

class SAC_Agent():
    def __init__(self, args, 
                 ) -> None:
         
        self.get_config(args)
        
        actor_lr = self.LR
        critic_lr = self.LR
        alpha_lr = self.LR 
        activation = 'relu'
        self.auto_temp = False
        self.prioreplay = False
        
        self.Transition = namedtuple('Transition',
                ('state','action','next_state','reward', 'done'))        
        if self.prioreplay: 
            self.memory = PrioritizedReplayBuffer(
                alpha=args.buffer_alpha,   #  alpha 0 would equal uniform distribution 
                beta=args.buffer_beta, 
                storage=ListStorage(max_size=self.MEMORY_SIZE)
            )
        else:
            self.memory =  ReplayBuffer(
                storage=ListStorage(max_size=self.MEMORY_SIZE)
            )
        
        self.actor = ActorNetwork(lr=actor_lr, 
                                  conv_arch=self.conv_arch,
                                  activation=activation,
                                  name="actor"
                                  )
        self.critic_local = CriticNetwork(critic_lr,
                                      self.conv_arch,
                                      activation=activation,
                                      name="critic_local")
        self.critic_local_2 = CriticNetwork(critic_lr,
                                      self.conv_arch,
                                      activation=activation,
                                      name="crictic_local_2")

        self.critic_target = CriticNetwork(critic_lr,
                                      self.conv_arch,
                                      activation=activation,
                                      name="critic_target")
        self.critic_target_2 = CriticNetwork(critic_lr,
                                      self.conv_arch,
                                      activation=activation,
                                      name="critic_target_2")
        
        copy_params(self.critic_local, self.critic_target)
        copy_params(self.critic_local_2, self.critic_target_2)
        
        if self.auto_temp: 
            self.target_entropy = 1     # |dimA|
            self.init_entropy = 5.0
            self.log_alpha = torch.tensor(self.init_entropy,dtype=torch.float64, requires_grad=True, device=self.actor.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        else: 
            self.alpha = np.exp(self.log_alpha)

    def get_config(self, args):
        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.gamma
        self.TAU = args.tau
        self.LR = args.lr
        self.num_episodes = args.num_episodes
        self.MEMORY_SIZE = args.memory_size
        self.conv_arch = args.conv_arch
        self.tau = args.tau
        self.log_alpha = args.log_alpha
    
    def select_action(self, state, greedy = False ):
        # state = torch.Tensor([observation]).to(self.actor.device)
        self.actor.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)
            sampled_action, action_probs, log_probs = self.actor.sample_dist(state)
        if greedy:
        #TODO: look here maybe not yeild correctly 
            choice = action_probs.argmax()
        else: 
            choice = sampled_action
        self.actor.train()
        return choice
      
    def save_experience(self, state, action, next_state, reward, done):
        '''Transition should be a list of [state, action, next state, reward]'''
        
        transition = self.Transition(state, action, next_state, reward,done)
        self.memory.add(transition)
        
    def sample_from_experience(self):
        '''Sample from the experience replay buffer
        Return them as batches of tensor ready to be optimized
        '''
        # get memory statistic (this is too slow)
        # This first line give reward tuple of tensors
        # reward_list  = torch.stack(list(zip(*self.memory._storage._storage))[3], dim=0)
        # whole_storage = self.memory._storage._storage
        # whole_reward = []
        # for item in whole_storage: 
        #     whole_reward.append(item.reward.item())
        # reward_mean = np.mean(whole_reward)
        # reward_std = np.std(whole_reward)

        
        samples, info = self.memory.sample(batch_size=self.BATCH_SIZE,return_info=True)
        
        state_batch = samples.state
        action_batch = samples.action
        reward_batch = samples.reward
        next_state_batch = samples.next_state
        done_batch = samples.done
        
        if not self.prioreplay:
            weights = None
            indexes = None
        else:
            weights = torch.tensor(info['_weight'], dtype=torch.float64, device= self.actor.device)
            indexes = info['index'] 
        
        return state_batch, action_batch, reward_batch, next_state_batch,done_batch, weights, indexes
        
        
    def soft_update(self, source_net, target_net, tau = 0.05):
        """This update is a soft update"""
        
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        if self.memory.__len__() < self.BATCH_SIZE: 
                return None

        # states_ is the next state
        states, actions, rewards, states_, dones, weights, indexes = self.sample_from_experience()
        
        # Calculate Critic Losses 
        with torch.no_grad(): 
            # Grab action from actor current policy 
            _,  actions_probs, log_probs = self.actor.sample_dist(states_)
            # Get actual value of the next states
            q1_next_target = self.critic_target(states_)
            q2_next_target = self.critic_target_2(states_)
            q_target_ = torch.min(q1_next_target, q2_next_target)
            min_q_target_ = actions_probs * (q_target_ - self.alpha * log_probs)
            min_q_target_ = min_q_target_.sum(dim=1)
            next_q_value = rewards + (1 - dones) * self.GAMMA * min_q_target_
        
        qf1 = self.critic_local(states).gather(1, actions).view(-1)
        qf2 = self.critic_local_2(states).gather(1, actions).view(-1)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        if self.prioreplay: 
            td_error1 = (qf1 - next_q_value).clone().detach()
            td_error2 = (qf2 - next_q_value).clone().detach()
            td_error = torch.min(td_error1, td_error2)
            qf1_loss *= torch.mean(weights)
            qf2_loss *= torch.mean(weights)
            self.memory.update_priority(indexes,
                                        priority=(abs(td_error) + 0.00001)) 
        self.critic_local.optimizer.zero_grad()
        self.critic_local_2.optimizer.zero_grad()
        qf1_loss.backward(retain_graph=False)
        qf2_loss.backward(retain_graph=False)
        self.critic_local.optimizer.step()
        self.critic_local_2.optimizer.step()
        self.soft_update(source_net = self.critic_local,
                         target_net= self.critic_target, tau=self.TAU)
        self.soft_update(source_net= self.critic_local_2, 
                         target_net= self.critic_target_2, tau=self.TAU)

        # Calculate actor losses 
        _, actions_probs, log_probs = self.actor.sample_dist(states)
        with torch.no_grad():
            qf1_pi = self.critic_local(states)
            qf2_pi = self.critic_local_2(states)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = self.alpha * log_probs - min_qf_pi
        policy_loss = (actions_probs * policy_loss).sum(dim=1).mean()
        self.actor.optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.actor.optimizer.step()

        if self.auto_temp:
            log_pi = torch.sum(log_probs.detach() * actions_probs.detach(), dim=1)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy))
            alpha_loss = alpha_loss.mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward(retain_graph=False)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        # alpha : Temperature loss
        loss_dict = {
                "critic loss" : qf1_loss.item(), 
                "actor loss" : policy_loss.item(), 
                # "value loss" : value_loss,
                # "alpha loss" : alpha_loss.item(),
                "alpha": self.alpha
        }

        return loss_dict
        
    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()


def copy_params(from_model, to_model):
    for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
        to_model.data.copy_(from_model.data.clone())