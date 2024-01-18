from collections import deque, namedtuple
from copy import deepcopy
from .net import fc, conv_net, duel_net
from torchrl.data import ListStorage, PrioritizedReplayBuffer
import random, math, torch 
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DQN_Agent_PER():
    """Implement simple DQN with replay buffer."""

    def __init__(self, args,
                 num_inputs, 
                 num_outputs, 
                 device, 
                 layer_dims=[128,128], 
                 activation='relu', 
                 conv_architecture=None, 
                 weight_init=True,
                 pre_trained_model='none',
                 variable_lr=True, 
                 dropout_rate=0.0,
                 net_type='conv',
                 **kwargs):
        super(DQN_Agent_PER, self).__init__(device)

        self.get_config(args)
        self.num_inputs = num_inputs

        if net_type=='fc':
            self.policy = fc(num_inputs, num_outputs, layer_dims, device, activation)
            self.target = fc(num_inputs, num_outputs, layer_dims, device, activation)

        elif net_type=='conv':
            self.policy = conv_net(conv_architecture, device, activation, dropout_rate=dropout_rate)
            self.target = conv_net(conv_architecture, device, activation, dropout_rate=dropout_rate)

        elif net_type=='duel':
            self.policy = duel_net(conv_architecture, activation)
            self.target = duel_net(conv_architecture, activation)

        else:
            print('Unrecognized type of neural network')


        # weight initialization
        if weight_init:
            self.init_weights()

        # Copy the main network to target network 
        if pre_trained_model == 'none':
            self.target.load_state_dict(deepcopy(self.policy.state_dict()))
        # load a pretrained model
        else:
            saved_model = torch.load(pre_trained_model)
            self.target.load_state_dict(deepcopy(saved_model))
            self.policy.load_state_dict(deepcopy(saved_model))


        self.Transition = namedtuple('Transition',
                ('state','action','next_state','reward', 'done'))

        self.memory = PrioritizedReplayBuffer(
            alpha=args.buffer_alpha,   #  alpha 0 would equal uniform distribution 
            beta=args.buffer_beta, 
            storage=ListStorage(args.memory_size),
            batch_size=args.batch_size
        )
        
        self.steps_done=0
        self.device = device

        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                     lr = self.LR, 
                                     amsgrad=True)
        
        # self.criterion = nn.MSELoss()
        self.criterion = nn.SmoothL1Loss()
        
        if variable_lr:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                        patience=1, threshold=0.001, threshold_mode='rel', 
                                        cooldown=0, min_lr=0, eps=1e-08, verbose=False)
      
      
    def init_weights(self):
        '''Implement Xavier weight initialization'''
        for key in self.policy.state_dict():
            if 'weight' in key:
                torch.nn.init.xavier_uniform(self.policy.state_dict()[key], gain=nn.init.calculate_gain('relu'))


    def get_config(self,args):
        '''PORT Config variable from cfg.py to agent'''

        self.BATCH_SIZE = args.batch_size
        self.GAMMA = args.gamma
        self.EPS_START = args.eps_start
        self.EPS_END = args.eps_end
        self.EPS_DECAY = args.eps_decay
        self.TAU = args.tau
        self.LR = args.lr
        self.num_episodes = args.num_episodes
        self.MEMORY_SIZE = args.memory_size
    
    def select_action(self, state, greedy=False):
        '''Action selection based on decay greedy.
        Set greedy to True when running validation and testing 
        '''
        sample = random.random()
        if greedy: 
            eps_threshold = 0
        else:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
        
        # Turn the grad on of incase batchnorm and dropout layers
        self.policy.eval()
        with torch.no_grad():
            action_chosen = self.policy(state).argmax()
        self.policy.train()
        
        
        if sample < eps_threshold:
            action_space = [i for i in range(5) if i != action_chosen.item()] # greedy is picked among the least likely actions
            action_chosen = torch.tensor(random.sample(action_space,1)[0], device=self.device, dtype=torch.long)
        
        return action_chosen

    def optimize(self,step_done):
        '''Run optimization step'''

        if step_done <= self.MEMORY_SIZE:
            # Filling the buffer before start training
            return None
        
        # Sample from the experience replace 
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indexes = self.sample_from_experience(self.BATCH_SIZE)
        # Compute Q(s_t, a)
        state_action_values = self.policy(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze()
        
        # Compute V for all next states
        with torch.no_grad(): 
            next_state_values = self.target(next_state_batch).max(1)[0]
            
        expected_state_action_values = reward_batch + next_state_values * self.GAMMA * (1-done_batch) 
        
        # TD error 
        td_error = state_action_values - expected_state_action_values
        
        # Compute Loss
        loss = self.criterion(state_action_values, expected_state_action_values)
        # The loss is modulated by the weights of the prioritized replay 
        # to stablize the training 
        # This is impoartance sampling. but we do mean because we're looking at batch 
        # The weights is also linear so we can add it here
        loss *= torch.mean(weights)
        
        # Optimize the model 
        self.optimizer.zero_grad()
        loss.backward()
        
        # Inplace gradient clipping 
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.optimizer.step()
        
        self.memory.update_priority(indexes, 
                                    priority= (abs(td_error)+0.001))
        return loss.item()

    #### EXPERIENCE REPLAY
    def save_experience(self, state, action, next_state, reward, done):
        '''Transition should be a list of [state, action, next state, reward]'''
        
        transition = self.Transition(state, action, next_state, reward,done)
        self.memory.add(transition)
        

    def sample_from_experience(self, sample_size):
        '''Sample from the experience replay buffer
        Return them as batches of tensor ready to be optimized
        '''
        samples, info = self.memory.sample(batch_size=self.BATCH_SIZE,return_info=True)
        
        state_batch = samples.state.squeeze()
        action_batch = samples.action.squeeze()
        reward_batch = samples.reward.squeeze()
        next_state_batch = samples.next_state.squeeze()
        done_batch = samples.done
        weights = torch.tensor(info['_weight'], dtype=torch.float64, device= self.device)
        indexes = info['index'] 
        
        
        return state_batch, action_batch, reward_batch, next_state_batch,done_batch, weights, indexes
        