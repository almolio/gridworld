# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random, torch
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np


class Environment(object):
    def __init__(self, 
                 variant, 
                 data_dir, 
                 neural_net_type='fc',
                 obs_model='std'):
        
        # observation model
        self.neural_net_type = neural_net_type
        self.agent_loc_history = np.zeros([5,5], dtype=np.dtype('float32'))
        self.obs_model=obs_model

        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir

        self.training_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/training_episodes.csv')
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/validation_episodes.csv')
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f'/variant_{self.variant}/test_episodes.csv')
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0
        self.last_action = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3
        self.obs_chan0 = np.zeros([5,5])
        if self.variant == 0 or self.variant == 1:
            self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
                                   (1,0), (1,1), (1,2), (1,3), (1,4),
                                   (2,0), (2,1), (2,2), (2,3), (2,4),
                                   (3,0), (3,1), (3,2), (3,3), (3,4),
                                   (4,0), (4,1), (4,2), (4,3), (4,4)]
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]
        for cell in self.eligible_cells:
            self.obs_chan0[cell[0],cell[1]] = 1.0 # cells available encoded with 1
        self.obs_chan0[self.target_loc[0],self.target_loc[1]] = -1.0

        # distionary of observation models ordered for each variant
        self.obs_model_dict = {0: {'std': self.get_feature_image_6_chan_var0(),
                               'reachability': self.get_feature_image_6_chan_reachability()},
                               1: {'std': self.get_feature_image_6_chan_var1()},
                               2: {'std': self.get_feature_image_6_chan_reachability()}
                               }

    def reset(self, mode):
        modes = ['training', 'validation', 'testing']
        if mode not in modes:
            raise ValueError('Invalid mode. Expected one of: %s' % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        if mode == "testing":
            try:
                episode = self.test_episodes[0]
                self.test_episodes.remove(episode)
            except Exception as e:
                print(e)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f'/variant_{self.variant}/episode_data/episode_{episode:03d}.csv',
                                index_col=0)

        return self.get_obs()

    def get_entire_episode(self):
        '''Query the content of the episode'''
        return self.data.to_numpy(copy=True)

    def get_state(self):
        """Return the state of the environment as a 5x5 numpy array with following codes: 
            background_value = 0
            target_value = 1
            item_value = 2
            agent_value = 3
            agent on target location = 4
            agent loaded with item = 5
        """ 
        state = np.zeros([5,5])
        state[self.vertical_idx_target][self.horizontal_idx_target] = 3 # target
        if self.agent_loc==self.target_loc:
            state[self.agent_loc[0]][self.agent_loc[1]] = 4 # agent on target cell
        elif self.agent_load:
            state[self.agent_loc[0]][self.agent_loc[1]] = 5 # agent loaded with at least one item
        else:
            state[self.agent_loc[0]][self.agent_loc[1]] = 1 # agent alone in cell
        for item in self.item_locs:
            state[item[0]][item[1]] = 2 # item
        return state

    def step(self, act):
        """Take one environment step based on the action"""
        self.step_count += 1
        self.last_action = act
        rew = 0

        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            if new_loc in self.eligible_cells:
                self.agent_loc = deepcopy(new_loc)
                rew += -1

            # keep track of agent's trajectory in the environment
        self.agent_loc_history[:,:] -= 0.2
        self.agent_loc_history[self.agent_loc_history<0.1] = 0.1
        self.agent_loc_history[self.agent_loc[0],self.agent_loc[1]] = 1

        # item pick-up
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
                self.agent_load += 1
                idx = self.item_locs.index(self.agent_loc)
                self.item_locs.pop(idx)
                self.item_times.pop(idx)
                rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_obs()

        return rew, next_obs, done

    def get_obs(self):
        '''Returns observations either as a vector of features for fully connected network
        or as an image with i channels for conv network'''
        
        # call update 
        
        obs = self.get_feature_image_6_chan_var0()
        
        # obs = self.obs_model_dict[self.variant][self.obs_model]
        # print(obs)
        return obs

    def get_feature_image_6_chan_reachability(self):
        '''Compute and return an image with i channels based on a given encoding of the env state
        Channels 5,6 encode locations of items within reach and time buffer to pick them up'''
        # init
        obs = np.ones([6,5,5], dtype=np.dtype('float32')) * 0.1
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, 
        # channel 2 for available capacity, channel 3 for load
        # channel 4 for items, channel 5 for item time counters  
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[1,agent_loc[0],agent_loc[1]] = 1.0
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity
        obs[3,agent_loc[0],agent_loc[1]] = 0.0 if (self.agent_load > 0.0) else 1.0

        # if there is any item
        if self.item_locs:
            item_locs = np.array(self.item_locs)
            item_distances = self.compute_agent_item_distances(type='manhattan')
            reachability_channel = np.ones([5,5], dtype=np.dtype('float32')) * 0.1

            for item_idx in range(len(self.item_locs)):
                # store item locations and reachability index
                reachability_idx = (10-self.item_times[item_idx]-1) - item_distances[item_idx] 
                if reachability_idx > 0:
                    obs[5,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                    reachability_channel[item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx]) - item_distances[item_idx] 
            
            reachability_channel[reachability_channel>0.1] /= np.max(reachability_channel)
            obs[5,:,:] = reachability_channel
        
        return torch.from_numpy(obs)

    def get_feature_image_6_chan_w_distances(self):
        '''Variation added: encoding of the item location is normalized with Manhattan distance'''
        # init
        obs = np.ones([6,5,5], dtype=np.dtype('float32')) * 0.1
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, 
        # channel 2 for available capacity, channel 3 for load
        # channel 4 for items, channel 5 for item time counters  
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[1,agent_loc[0],agent_loc[1]] = 1.0
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity
        obs[3,agent_loc[0],agent_loc[1]] = 0.0 if (self.agent_load > 0.0) else 1.0

        # if there is any item
        if self.item_locs:
            # make a list of item distance 
            item_locs = np.array(self.item_locs)
            item_distances = []
            
            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long they've been in the environment
                obs[4,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                distance = np.linalg.norm([agent_loc,item_locs[item_idx, :]])
                obs[5,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10

        return torch.from_numpy(obs)

    def get_feature_image_6_chan_var1(self):
        '''Variation added: encoding of the load is normalized with capacity'''
        # init
        obs = np.ones([6,5,5], dtype=np.dtype('float32')) * 0.1
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, 
        # channel 2 for available capacity, channel 3 for load
        # channel 4 for items, channel 5 for item time counters  
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[1,agent_loc[0],agent_loc[1]] = 1.0
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity
        obs[3,agent_loc[0],agent_loc[1]] = 1.0 - (self.agent_load / self.agent_capacity)

        # if there is any item
        if self.item_locs:
            item_locs = np.array(self.item_locs)
            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long they've been in the environment
                obs[4,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                obs[5,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10
        
        return torch.from_numpy(obs)
    
    def get_feature_image_6_chan_var0(self):
        '''Compute and return an image with i channels based on a given encoding of the env state'''
        # init
        obs = np.ones([6,5,5], dtype=np.dtype('float32')) * 0.1
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, 
        # channel 2 for available capacity, channel 3 for load
        # channel 4 for items, channel 5 for item time counters  
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[1,agent_loc[0],agent_loc[1]] = 1.0
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity
        obs[3,agent_loc[0],agent_loc[1]] = 0.0 if (self.agent_load > 0.0) else 1.0

        # if there is any item
        if self.item_locs:
            # make a list of item distance 
            item_locs = np.array(self.item_locs)

            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long they've been in the environment
                obs[4,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                obs[5,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10
        
        return torch.from_numpy(obs)

    def get_feature_image_3_chan(self):
        '''Compute and return an image with i channels based on a given encoding of the env state'''
        # init
        obs = np.zeros([3,5,5], dtype=np.dtype('float32'))
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 2 for agent, channel 1 for items
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[2,agent_loc[0],agent_loc[1]] = 1.0
        if self.agent_load > 0: # should be adapted when working with the other variants
            obs[2,agent_loc[0],agent_loc[1]] = -1.0

        # if there is item
        if self.item_locs:
            # make a list of item distance 
            item_locs = np.array(self.item_locs)

            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long theyve been in the environment
                obs[1,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10
        
        return torch.from_numpy(obs)

    def get_feature_image_5_chan(self):
        '''Compute and return an image with i channels based on a given encoding of the env state'''
        # init
        obs = np.zeros([5,5,5], dtype=np.dtype('float32'))
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, channel 2 for available capacity, 
        # channel 3 for items, channel 4 for item time counters 
        obs[0,target_loc[0],target_loc[1]] = 1.0
        obs[1,agent_loc[0],agent_loc[1]] = 1.0
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity

        # if there is item
        if self.item_locs:
            # make a list of item distance 
            item_locs = np.array(self.item_locs)

            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long theyve been in the environment
                obs[3,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                obs[4,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10
        
        return torch.from_numpy(obs)

    def get_feature_image_6_chan_w_history(self):
        '''Variation added: agent location channels encodes trajectory of previous steps 
        decayed with -0.1 at each time step + encoding of the non available cells (if variant 2)'''
        # init
        obs = np.zeros([6,5,5], dtype=np.dtype('float32'))
        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)

        # channel 0 for target, channel 1 for agent, 
        # channel 2 for available capacity, channel 3 for load
        # channel 4 for items, channel 5 for item time counters  
        obs[0,:,:] = deepcopy(self.obs_chan0)
        obs[0,target_loc[0],target_loc[1]] = -1.0
        obs[1,:,:] = deepcopy(self.agent_loc_history)
        obs[2,agent_loc[0],agent_loc[1]] = (self.agent_capacity - self.agent_load)/self.agent_capacity
        obs[3,agent_loc[0],agent_loc[1]] = 0.0 if (self.agent_load > 0.0) else 1.0

        # if there is any item
        if self.item_locs:
            # make a list of item distance 
            item_locs = np.array(self.item_locs)

            for item_idx in range(item_locs.shape[0]):
                # decay of items based on how long they've been in the environment
                obs[4,item_locs[item_idx][0], item_locs[item_idx][1]] = 1
                obs[5,item_locs[item_idx][0], item_locs[item_idx][1]] = (10-self.item_times[item_idx])/10
        
        return torch.from_numpy(obs)

    def compute_agent_item_distances(self, type='manhattan'):
        '''Compute either manhattan or euclidian disntance to each item.'''
        agent_loc = np.array(self.agent_loc)
        # make a list of item distance 
        item_locs = np.array(self.item_locs)
        item_distances = []
        num_of_items = item_locs.shape[0]

        if type=='manhattan':
            for item_idx in range(num_of_items):
                # Find Euclidian distance
                dist2item=np.sum(abs(agent_loc - item_locs[item_idx, :]))
                item_distances.append(dist2item)
        
        elif type=='euclidian':
            for item_idx in range(num_of_items):
                # Find Euclidian distance
                distance = np.linalg.norm([agent_loc,item_locs[item_idx, :]])
                item_distances.append(distance)
        return item_distances

    def get_feature_vector(self):
        '''Compute and return a vector of features based on the given observation model'''

        agent_loc = np.array(self.agent_loc)
        target_loc = np.array(self.target_loc)
        
        # if there is item
        if self.item_locs:
            agent_to_target = agent_loc - target_loc

            # make a list of item distance 
            item_locs = np.array(self.item_locs)
            item_distances = []
            num_of_items = item_locs.shape[0]
            for item_idx in range(num_of_items):
                # Find Euclidian distance
                distance = np.linalg.norm([agent_loc,item_locs[item_idx, :]])
                item_distances.append(distance)
            closest_item_idx = np.argmin(item_distances)
            
            agent_to_item = agent_loc - item_locs[closest_item_idx]
            item_to_target = item_locs[closest_item_idx] - target_loc
            item_time = self.item_times[closest_item_idx]
            item_loc = self.item_locs[closest_item_idx]
        else: 
            agent_to_item = [0,0]
            item_to_target = [0,0]
            item_time = 0
            item_loc = [-4,-4]
        
        obs_returned = [(item_loc[0])/4, # normalization 
                        (item_loc[1])/4,
                        (agent_loc[0])/4,
                        (agent_loc[1])/4, 
                        self.target_loc[0]/4,
                        self.target_loc[1]/4,
                        (agent_to_item[1] + item_to_target[1])/4,
                        (agent_to_item[0] + item_to_target[0])/4,
                        self.agent_load,
                        item_time/10,
                        self.last_action/4
                        ] 

        return torch.tensor(obs_returned, dtype=torch.float32)
