#!/usr/bin/env python3

import random, torch, os, time
from datetime import datetime
import numpy as np
from envs.environment_t import Environment 
from agent_policy.policy_base_class import Policy, DQN_Agent
from agent_policy.policy_sac import SAC_Agent
from agent_policy.train_policy_cfg import DQNCfg
from utils.helper_fcts import class_to_dict, parse_arguments
from agent_policy.train import *
import sys
from agent_policy.train_sac import train_sac, savecheckpoint

## SELECT FOLLOWING PARAMETERS ##

mode = 'training' # TODO: implement testing mode
policy_name = 'sac'
neural_net_type = 'duel' # choose between 'conv' and 'fc' and 'duel'
device="cuda"
# pre_trained_model = None # leave to None if we're not doing transfert learning
### problem variant
variant = 1

ts = time.time()
date_time = datetime.fromtimestamp(ts)
str_date_time = date_time.strftime("%d-%m-%Y-%H-%M-%S")
wandb_run_name = 'BIG_SAC_Conv1_1_4' + neural_net_type + '_var_' + str(variant) + '_' + str(str_date_time)
log_results_wandb = True #True #False
os.mkdir(os.path.join('model',str_date_time)) 
save_dir = os.path.join((os.path.dirname(__file__)), "model", str(str_date_time), f"{wandb_run_name}")
save_dir_extension = ".pt"

## END OF PARAMETERS ##

# other parameters 
data_dir = './data'
policy_dict = {"dqn": 'DQN_Agent', "sac": 'SAC_Agent'} 
variable_lr = False # schedule for the learning rate

### seeding ###
seed = 420  # set seed to allow for reproducibility of results
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed=seed)

def run(args):
    '''Main loop.'''

    try:
        # train(args, env, agent, device, wandb_run_name, 
        #         log_results_wandb, variable_lr, save_dir, save_dir_extension)
        train_sac(args, env, agent, device, 
                  wandb_run_name=wandb_run_name,
                  save_dir=save_dir, 
                  log_results_wandb=log_results_wandb
        )
        savecheckpoint(agent,save_dir, name="end")
        
    except KeyboardInterrupt:
        print('***** ***** training interrupted! ***** *****')
        # Save the model that matches wandb naming covention 
        savecheckpoint(agent,save_dir, name="earlystop")


if __name__ == "__main__":
    print("hello, gridworld!")
    args = parse_arguments(sys.argv[1:])
    obs_model = args.obs_model
    print(f'training params {args}')
    if neural_net_type == 'fc':
        layers_dims = [args.layer_dim for i in range(args.num_hidden_layers)]
        conv_architecture=None
        env = Environment(variant, 
                          data_dir, 
                          neural_net_type=neural_net_type,
                        #   obs_model=obs_model
                          )
        state = env.reset(mode)
        n_obs = (1,len(state))

    if neural_net_type == 'conv' or neural_net_type == 'duel':
        layers_dims = None
        conv_architecture = args.conv_arch
        n_obs = (1, conv_architecture['num_channel'], 5, 5)
        env = Environment(variant, 
                          data_dir, 
                          neural_net_type=neural_net_type, 
                        #   obs_model=obs_model
                          )
        state = env.reset(mode)
        
    policy_class = eval(policy_dict[policy_name])
    agent: Policy = policy_class(args, 
                                #  num_inputs=n_obs, 
                                #  num_outputs=args.n_actions,
                                #  device=device,
                                #  layer_dims=layers_dims,
                                #  activation='relu',  
                                #  weight_init=True,
                                # #  pre_trained_model=pre_trained_model,
                                #  conv_architecture=conv_architecture,
                                #  variable_lr=False,
                                #  dropout_rate=args.dropout_rate,
                                #  net_type=neural_net_type,
                                 )
    run(args)

