#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('../envs')
from environment_t import Environment 

## SELECT FOLLOWING PARAMETERS ##
mode = 'training' 
nb_episodes = {'training':800, 'validation':100, 'testing':100}
variant = 0  
data_dir = './data'

def path_optimizer(agent_loc, target_loc, episode_data=None, variant=0):
    '''Returns a matrix dim 200x2 tracing a path of the agent's ideal trajectory following a dummy heuristics
    -> Go fetch the next closest item in the list provided it will still be there by the time you get there <-'''

    # init
    optimized_path = np.zeros([200,2])
    optimized_path[0,:] = agent_loc
    timestep=0
    tot_reward=0.0
    reward = 15.0
    lifetime = 9
    done = False

    # compute objects' total lifetime in gridworld
    if (episode_data[:,1:3]>199).any():
        episode_data[episode_data[:,0]>199] = 199
 
    # compute the distance to each item and to the target
    dist_agent_item = np.zeros([len(episode_data),5])
    dist_agent_item[:,0] = episode_data[:,0]
    dist_agent_item[:,1:3] = episode_data[:,1:3] - target_loc
    dist_agent_item[:,3:5] = target_loc - episode_data[:,1:3]

    for item_idx in range(len(dist_agent_item)):
        if item_idx==0:
            delta_to_next = episode_data[item_idx,1:3] - agent_loc
            item_loc = episode_data[item_idx,1:3]

        else:
            delta_to_next = dist_agent_item[item_idx,1:3]
            item_loc = episode_data[item_idx,1:3]
        dist_to_next=np.sum(abs(delta_to_next))

        if dist_to_next>(dist_agent_item[item_idx,0] + lifetime):
            continue

        ### FETCH ITEM
        else:
            # print("******** next item pos : ", item_loc)
            deltax = int(abs(delta_to_next[0]))
            deltay = int(abs(delta_to_next[1]))
            wait_time = episode_data[item_idx,0] - timestep - deltax - deltay

            # 2-step look-ahead to see whether another item might be more profitable
            if wait_time>0 and item_idx < (len(dist_agent_item)-1):
                tot_num_steps = episode_data[item_idx,0] - timestep + np.sum(abs(dist_agent_item[item_idx,3:5]))
                tot_num_steps_next = episode_data[item_idx+1,0] - timestep + np.sum(abs(dist_agent_item[item_idx+1,3:5]))

                if tot_num_steps_next < tot_num_steps:
                    continue

                elif item_idx <(len(dist_agent_item)-2):
                    tot_num_steps_next_next = episode_data[item_idx+2,0] - timestep + np.sum(abs(dist_agent_item[item_idx+2,3:5]))
                    if tot_num_steps_next_next < tot_num_steps:
                        continue

            # move horizontally
            # print("current pos : ", optimized_path[timestep,:])
            if deltax>0:
                for step in range(1,deltax+1):
                    timestep+=1
                    if timestep>199:
                        done=True
                        break
                    optimized_path[timestep,0] = optimized_path[timestep-1,0] + delta_to_next[0]/deltax
                    optimized_path[timestep,1] = optimized_path[timestep-1,1] 
                    # print("next pos : ", optimized_path[timestep,:])
                dist_agent_item[:,0] -= (step)

            # move vertically
            if deltay>0:
                for step in range(1,deltay+1):
                    timestep+=1
                    if timestep>199:
                        done=True
                        break
                    optimized_path[timestep,1] = optimized_path[timestep-1,1] + delta_to_next[1]/deltay
                    optimized_path[timestep,0] = optimized_path[timestep-1,0] 
                    # print("next pos : ", optimized_path[timestep,:])
                dist_agent_item[:,0] -= (step)
            tot_reward += (reward/2 - 0.1*(deltay + deltax))

            ### WAIT FOR ITEM TO GET DROPPED
            if wait_time>0.0:
                for _ in range(int(wait_time)):
                    timestep+=1
                    if timestep>199:
                        done=True
                        break
                    optimized_path[timestep,:] = optimized_path[timestep-1,:] 

            ### BRING ITEM TO BASE
            delta_to_next = dist_agent_item[item_idx,3:5]
            deltax = int(abs(delta_to_next[0]))
            deltay = int(abs(delta_to_next[1]))

            # move horizontally
            # print("current pos : ", optimized_path[timestep,:])
            if deltax>0:
                for step in range(1,deltax+1):
                    timestep+=1
                    if timestep>199:
                        done=True
                        break
                    optimized_path[timestep,0] = optimized_path[timestep-1,0] + delta_to_next[0]/deltax
                    optimized_path[timestep,1] = optimized_path[timestep-1,1] 
                    # print("next pos : ", optimized_path[timestep,:])
                dist_agent_item[:,0] -= (step)

            # move vertically
            if deltay>0:
                for step in range(1,deltay+1):
                    timestep+=1
                    if timestep>199:
                        done=True
                        break
                    optimized_path[timestep,1] = optimized_path[timestep-1,1] + delta_to_next[1]/deltay
                    optimized_path[timestep,0] = optimized_path[timestep-1,0] 
                    # print("next pos : ", optimized_path[timestep,:])
                dist_agent_item[:,0] -= (step)
            tot_reward += (reward/2 - 0.1*(deltay + deltax))
            # print("tot_reward : ", tot_reward)

        if done:
            print("tot_reward : ", tot_reward)
            break

    return optimized_path, tot_reward

if __name__ == "__main__":
    print("hello, gridworld!")

    env = Environment(variant, data_dir, neural_net_type='conv')
    state = env.reset(mode)
    episode_data = np.array(env.get_entire_episode())
    agent_loc = np.array(env.agent_loc)
    target_loc = np.array(env.target_loc)
    tot_rew = 0

    for _ in range(nb_episodes[mode]):
        _, rew = path_optimizer(agent_loc, target_loc, episode_data)
        tot_rew += rew
        state = env.reset(mode)
        episode_data = np.array(env.get_entire_episode())
    
    print('average reward over all ' + str(nb_episodes[mode]) + ' episodes : ', round(tot_rew/nb_episodes[mode]))