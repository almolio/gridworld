import numpy as np
from collections import namedtuple

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_channel = input_shape[0]
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.next_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size,1))
        self.reward_memory = np.zeros(self.mem_size)
        self.done_memory = np.zeros(self.mem_size, dtype=np.int8)

        self.Transition = namedtuple('Transition',
                ('state','action','next_state','reward', 'done'))   

    def add(self, transition):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = transition.state.cpu().numpy()
        self.next_state_memory[index] = transition.next_state.cpu().numpy()
        self.action_memory[index] = transition.action.cpu().numpy()
        self.reward_memory[index] = transition.reward.cpu().numpy()
        self.done_memory[index] = transition.done.cpu().numpy()

        self.mem_cntr += 1

    def sample(self, batch_size, normalized=True, return_info=True):
          
        max_mem = min(self.mem_cntr, self.mem_size)

        # Place holder for prioritized
        weights = None
        indexes = np.random.choice(max_mem, batch_size)
        if normalized == False:
            states = self.state_memory[indexes]
            states_ = self.next_state_memory[indexes]
            actions = self.action_memory[indexes]
            rewards = self.reward_memory[indexes]
            dones = self.done_memory[indexes]
        else:
            # Normalize everything
            end  = min(self.mem_cntr , self.mem_size)
            
            mean_state = np.mean(self.state_memory[:end], axis=(0), keepdims=True)
            std_state = np.std(self.state_memory[:end], axis=(0), keepdims=True)
            # states = self.state_memory[indexes]
            states = (self.state_memory[indexes] - mean_state) / (std_state + 1e-8)
            # states = self.state_memory[indexes]
            # for channel in range(self.n_channel):
                # states[:,channel,:,:] = (states[:,channel, :, :]- mean_state[channel] ) / std_state[channel]
            
            mean_state_ = np.mean(self.next_state_memory[:end], axis=(0), keepdims=True)
            std_state_ = np.std(self.next_state_memory[:end], axis=(0), keepdims=True)
            # states_ = self.next_state_memory[indexes]            
            states_ = (self.next_state_memory[indexes] - mean_state_) / (std_state_ + 1e-8)
            # states_ = self.next_state_memory[indexes]
            # for channel in range(self.n_channel):
            #     states_[:,channel,:,:] = (states_[:,channel, :, :]- mean_state_[channel] ) / std_state_[channel]
            
            mean_reward = np.mean(self.reward_memory[:end])
            std_reward = np.std(self.reward_memory[:end])
            rewards = (self.reward_memory[indexes] -  mean_reward) / std_reward
            
            actions = self.action_memory[indexes]
            dones = self.done_memory[indexes]

        return [states, actions, rewards, states_, dones], [weights, indexes]
    
