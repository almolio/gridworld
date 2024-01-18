from IPython.display import clear_output
import time 
import matplotlib.pyplot as plt
import numpy as np
import wandb
import argparse
import json

def launch_wandb(args, wandb_run_name):
    '''Launch login to weights and biases.'''
    
    wandb.login(key="dfa65b8704706dec24d99ffaa32549665ca9715a")
    
    wandb.init(
        entity = "itsonlygrid",
        # set the wandb project where this run will be logged
        project="Warehouse Gridworld",
        # track hyperparameters and run metadata
        # unpack args into a dict
        config=vars(args),
        name=wandb_run_name
    )

    # This will log all the code in the folder and subfolder
    wandb.run.log_code(".")

def class_to_dict(obj) -> dict:
    '''Store all class variables into a dictionary.'''

    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or "physics_engine" in key:
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        elif isinstance(val, dict):
            element = val
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def parse_arguments(arguments):
    '''  Example usage
    args = parse_arguments()
    print(args.learning_rate)
    print(args.batch_size)
    print(args.num_epochs)
    print(args.dropout_rate)    
    '''
    parser = argparse.ArgumentParser(description='Deep Learning Hyperparameters')
    
    # Add your hyperparameters as arguments
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.98, help='Discount factor for rewards')
    parser.add_argument('--eps_start', type=float, default=0.99, help='Starting value of epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.01, help='Ending value of epsilon for exploration')
    parser.add_argument('--eps_decay', type=float, default=1e6, help='Decay rate of epsilon')
    parser.add_argument('--tau', type=float, default=1.0, help='Soft update coefficient for target network')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=1e3, help='Decay rate of learning rate')
    parser.add_argument('--num_episodes', type=int, default=12501, help='Number of episodes for training')
    parser.add_argument('--memory_size', type=int, default=15000, help='Size of the replay memory')
    parser.add_argument('--target_update_period', type=int, default=9000, help='Period of target network update')
    parser.add_argument('--episode_length', type=int, default=200, help='Maximum length of an episode')
    parser.add_argument('--val_check_period', type=int, default=150, help='Period of validation check')
    parser.add_argument('--val_nb_episodes', type=int, default=5, help='Number of episodes for validation')
    parser.add_argument('--conv_arch', type=json.loads, 
                        default={"stride" : "(1,1)", "num_channel": 6, 
                                 "kernel_size":"(3,3)", "out_num_actions" : 5, 
                                 "padding" : 1, 'cnn_layer_depths': "(16,16,32)"}, 
                                 help='Architecture of conv net')
    parser.add_argument('--pre_trained_model', type=str, default='none', help='pathname of pre-trained model')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='parameter of dropout layers')
    parser.add_argument('--net_type', type=str, default='conv', help='type of nn architecture')
    parser.add_argument('--obs_model', type=str, default='std', help='observation model')
    parser.add_argument('--double_dqn', type=bool, default=True, help='whether or not to use double dqn technique')
    parser.add_argument('--variable_lr', type=bool, default=True, help='scheduling of learning rate conditioned on increases in total val reward')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    
    ### ENV
    parser.add_argument('--variant', type=int, default=0, help='Problem variant')
    parser.add_argument('--gen_transition_prob', type=float, default=0.05, help='Probability of adding an imaginary transition to replay buffer')
    parser.add_argument('--imaginary_reward', type=float, default=1.0, help='Reward during imaginary transitions')
    parser.add_argument('--random_agent_loc', type=bool, default=True, help='Randomize agent location at beginning of episode')

    ### AGENT 
    parser.add_argument('--n_actions', type=int, default=5, help='Number of actions')
    parser.add_argument('--num_hidden_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--layer_dim', type=int, default=32, help='Dimension of each hidden layer')
    
    ### Memory
    parser.add_argument('--buffer_alpha', type=float, default=0.7, help='ultility of prioritization')
    parser.add_argument('--buffer_beta', type=float, default=0.5, help='Importance sampling weights')
    
    ### SAC Params
    parser.add_argument('--log_alpha', type=float, default=-1, help='entropy of the system, if autoalpha on, then the starting entropy')
    
    
    args = parser.parse_args(arguments)
    
    return args

def display_state(state, episode, episode_nr, time , total_rew):
    '''Display the environment state with color coding.'''
    
    color_map = {0: np.array([255, 255, 255]), # white
                 1: np.array([255, 0, 0]), # red
                 2: np.array([0, 0, 255]), # blue
                 3: np.array([0, 0, 0]), # black
                 4: np.array([0, 0, 0]), # black
                 5: np.array([255, 0, 0])} # red
                  
    txt_label = {0: '',
                 1: 'agent', 
                 2: 'item', 
                 3: 'target',
                 4: 'agt + tgt',
                 5: 'agt + item'} 
    
    txt_color = {0: 'black',
                 1: 'black', 
                 2: 'white', 
                 3: 'white',
                 4: 'red',
                 5: 'blue'} 
    
    data_3d = np.ndarray(shape=(state.shape[0], state.shape[1], 3), dtype=int)
    for i in range(0, state.shape[0]):
        for j in range(0, state.shape[1]):
            data_3d[i][j] = color_map[state[i][j]]
            
    # display the plot 
    fig, ax = plt.subplots(1,1)
    ax.imshow(data_3d)

    # add numbers to the plot 
    for i in range(0, state.shape[0]):
        for j in range(0, state.shape[1]):
            c = state[j,i]
            ax.text(i, j, txt_label[c], va='center', ha='center', fontsize=20, color = txt_color[c])

    ax.set_title('Mode ' + str(episode) + ' episode # ' + str(episode_nr) + f'_{time}_' + ' total rew: ' + str(total_rew), fontsize=20)
    ax.set_xlabel('x-horizontal', fontsize=16)
    ax.set_ylabel('y-vertical', fontsize=16)
    ax.figure.set_size_inches(10, 10)
    plt.show()