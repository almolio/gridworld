import subprocess

args_dict = {
    'batch_size': 256,
    'gamma': 0.98, # was 0.9
    # 'eps_start': 0.5,
    # 'eps_end': 0.5,
    # 'eps_decay': 1e6,
    'tau': 0.05, # was 0.005
    'lr': 2e-4,
    # 'lr_decay': 1e3,
    'num_episodes': 13000,
    'memory_size': 20000,
    # 'target_update_period': 1000,
    # 'episode_length': 200,    
    'val_check_period': 100,
    'val_nb_episodes': 10,
    # 'n_actions': 5,
    # 'num_hidden_layers': 5,
    # 'layer_dim': 32,
    # 'conv_arch': '{"stride" : "(1,1)", "num_channel": 3, "kernel_size":"(3,3)", "out_features" : 5, "padding" : 1}'
    'log_alpha': -0.5
        
}
### TESTING DROPOUT ARCHITECT + SMALL LR
### TEsting small memory
#### Testing no drop out in covn net 



command = ['python', 'main.py']
for key, value in args_dict.items():
    command.extend(['--' + key, str(value)])


# Execute the command
subprocess.run(command)
