import torch
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
import ast
from torch import optim
import os
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class duel_net(nn.Module):
    '''Construct the dueling net work for the agen '''

    def __init__(self, conv_architecture, activation='relu', dropout_rate=0.0):
        super(duel_net, self).__init__()
        
        stride=ast.literal_eval(conv_architecture['stride'])
        num_channel = conv_architecture['num_channel']
        kernel_size=ast.literal_eval(conv_architecture['kernel_size'])
        padding=conv_architecture['padding']
        out_num_actions=conv_architecture['out_num_actions']
        activation = get_activation(activation)
        cnn_layer_depths = ast.literal_eval(conv_architecture['cnn_layer_depths'])
        dropout_rate = dropout_rate
        nn_layers = []
        in_depth = 0
        layer_depth = 0
        
        for i, layer_depth in enumerate(cnn_layer_depths):
            if i ==0:
                ### FIRST LAYER
                # output size: (5 - 3 + 2 * 1 )/ 1 + 1 = 5
                nn_layers.append(nn.Conv2d(num_channel, 
                                        out_channels=layer_depth, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=padding, 
                                        dtype=torch.float32))
            if i ==1:
                ### SECOND LAYER
                # output size: (5 - 3 + 0 )/ 2 + 1 = 2
                nn_layers.append(nn.Conv2d(in_depth, 
                                        out_channels=layer_depth, 
                                        kernel_size=(3,3), 
                                        stride=(2,2),
                                        padding=0, 
                                        dtype=torch.float32))
            if i ==2:
                ### THIRD LAYER
                # output size: (2 - 2 + 0 )/ 1 + 1 = 1
                nn_layers.append(nn.Conv2d(in_depth, 
                                        out_channels=layer_depth, 
                                        kernel_size=(2,2), 
                                        stride=(1,1),
                                        padding=0, 
                                        dtype=torch.float32))
                
            nn_layers.append(activation)
            in_depth = layer_depth
            if dropout_rate>0.0:
                nn_layers.append(nn.Dropout(p=dropout_rate))

        nn_layers.append(nn.Flatten(start_dim=1, end_dim=-1))   
        
        self.feature_layer = nn.Sequential(*nn_layers).to(device)
        
        self.value_stream = nn.Sequential(
            nn.Linear(in_depth, int(in_depth/2)),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(int(in_depth/2), 32),
            activation,
            nn.Linear(32, 1)
        ).to(device)     

        self.advantage_stream = nn.Sequential(
            nn.Linear(in_depth, int(in_depth/2)),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(int(in_depth/2), 32),
            activation,
            nn.Linear(32, out_num_actions)
        ).to(device)      
        
    def forward(self, x):
        features= self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals
    
class conv_net(nn.Module):
    '''Construct the conv net work for the agent '''

    def __init__(self, conv_architecture, device='cuda', activation='relu', dropout_rate=0.0):
        '''Init a convolutional neural net.'''
        super(conv_net, self).__init__()
        
        stride=ast.literal_eval(conv_architecture['stride'])
        num_channel = conv_architecture['num_channel']
        kernel_size=ast.literal_eval(conv_architecture['kernel_size'])
        padding=conv_architecture['padding']
        out_num_actions=conv_architecture['out_num_actions']
        activation = get_activation(activation)
        cnn_layer_depths = ast.literal_eval(conv_architecture['cnn_layer_depths'])
        dropout_rate = dropout_rate
        nn_layers = []
        in_depth = 0
        layer_depth = 0
        for i, layer_depth in enumerate(cnn_layer_depths):
            if i ==0:
                ### FIRST LAYER
                # output size: (5 - 3 + 2 * 1 )/ 1 + 1 = 5
                nn_layers.append(nn.Conv2d(num_channel, 
                                        out_channels=layer_depth, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=padding, 
                                        dtype=torch.float32))
            if i ==1:
                ### SECOND LAYER
                # output size: (5 - 3 + 0 )/ 2 + 1 = 2
                nn_layers.append(nn.Conv2d(in_depth, 
                                        out_channels=layer_depth, 
                                        kernel_size=(3,3), 
                                        stride=(2,2),
                                        padding=0, 
                                        dtype=torch.float32))
            if i ==2:
                ### THIRD LAYER
                # output size: (2 - 2 + 0 )/ 1 + 1 = 1
                nn_layers.append(nn.Conv2d(in_depth, 
                                        out_channels=layer_depth, 
                                        kernel_size=(2,2), 
                                        stride=(1,1),
                                        padding=0, 
                                        dtype=torch.float32))
                
            nn_layers.append(activation)
            in_depth = layer_depth
            if dropout_rate>0.0:
                nn_layers.append(nn.Dropout(p=dropout_rate))

        nn_layers.append(nn.Flatten(start_dim=1, end_dim=-1))

        ### FOURTH LAYER
        nn_layers.append(nn.Linear(in_depth, int(in_depth/2), dtype=torch.float32))
        nn_layers.append(activation)

        ### FIFTH LAYER
        nn_layers.append(nn.Linear(int(in_depth/2), out_num_actions, dtype=torch.float32))
        nn_layers.append(activation)
        
        self.net = nn.Sequential(*nn_layers).to(device) 
        
    def forward(self, x):
        return self.net(x)

class fc(nn.Module):
    def __init__(self,num_inputs, num_outputs, layer_dims, device, activation):
        super(fc, self).__init__()
        
        '''Init a fully connected neural net.'''
        activation = get_activation(activation)
        nn_layers = []
        nn_layers.append(nn.Linear(num_inputs[1], layer_dims[0]))
        nn_layers.append(activation)

        for l in range(len(layer_dims)):
            if l == len(layer_dims) - 1:   # Last layer
                nn_layers.append(nn.Linear(layer_dims[l], num_outputs))
                nn_layers.append(activation)
            else:
                nn_layers.append(nn.Linear(layer_dims[l], layer_dims[l + 1]))
                nn_layers.append(activation)
                # nn_layers.append(nn.Dropout(p=0.5))
        # nn_layers.append(nn.Softmax(dim=0))  # TODO: make sure this is the dim for softmax
        
        self.net = nn.Sequential(*nn_layers).to(device) 
    
    def forward(self, x):
        return self.net(x)
        
def create_feature_conv(conv_architecture, activation):
    dropout_rate = 0.3
    activation = get_activation(activation)
    
    
    conv_layer0 = nn.Conv2d(in_channels=3,
                            out_channels= 32, 
                            kernel_size= (1,1),
                            stride= (1,1),
                            padding=0,
                            dtype=torch.float32)
    
    
    conv_layer1 = nn.Conv2d(in_channels=32,
                            out_channels= 64, 
                            kernel_size= (1,1),
                            stride= (1,1),
                            padding=1,
                            dtype=torch.float32)
    
    conv_layer2 = nn.Conv2d(in_channels=64,
                            out_channels= 32, 
                            kernel_size= (4,4),
                            stride= (1,1),
                            padding=1,
                            dtype=torch.float32)
    flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
    
    
    nn_layers = [
        # nn.BatchNorm2d(3, affine=False),
        conv_layer0, 
        # nn.BatchNorm2d(32, affine=False),
        activation,
        # nn.Dropout(0.5),
        conv_layer1, 
        # nn.BatchNorm2d(64, affine=False),
        activation, 
        # nn.Dropout(0.3),
        conv_layer2,
        activation,
        flatten_layer
                ]
    
    return nn.Sequential(*nn_layers).to(device)


class CriticNetwork(nn.Module):
    def __init__(self, lr, conv_architecture, activation, name='critic', chkpt_dir='temp/sac' ):
        super(CriticNetwork, self).__init__()
        
        
        self.name = name 
        self.chkpt_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.chkpt_dir, name+'_sac')
        self.activation = get_activation(activation)
        self.n_actions = conv_architecture['out_num_actions']
        
        # Creating layers for the network         
        self.feature_ext_block = create_feature_conv(conv_architecture, activation)
        self.action_value_stream = nn.Sequential(
            nn.Linear(1152, 512),
            self.activation, 
            # nn.Dropout(0.5),
            nn.Linear(512, 128),
            self.activation, 
            # nn.Dropout(0.3),
            nn.Linear(128,64),
            self.activation,
            nn.Linear(64,5)
        )
                
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        init_weights(self)
        
    def forward(self, state):
        state_features = self.feature_ext_block(state)
        action_value = self.action_value_stream(state_features)
        return action_value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoit(self): 
        self.load_state_dict(torch.load(self.checkpoint_file))
        
        
class ValueNetwork(nn.Module):
    def __init__(self,lr, conv_arch, activation = 'relu', name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.lr = lr 
        self.name = name 
        self.chckpt_dir = chkpt_dir 
        self.checkpoint_file = os.path.join(self.chckpt_dir, name+'_sac')
        self.activation = get_activation(activation)
        
        self.feature_ex_block = create_feature_conv(conv_arch, activation)
        
        self.value_stream = nn.Sequential(
            nn.Linear(32, 256),
            self.activation, 
            nn.Linear(256, 256),
            self.activation, 
            nn.Linear(256,1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch. device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device)
        init_weights(self)
        
    def forward(self, state):
        state_value = self.feature_ex_block(state)
        state_value = self.value_stream(state_value)
        
        return state_value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoit(self): 
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, lr, conv_arch, activation='lrelu', name='actor', chkpt_dir = 'temp/sac'):
        super(ActorNetwork, self).__init__()
        
        self.lr = lr 
        self.name = name 
        self.chckpt_dir = chkpt_dir 
        self.checkpoint_file = os.path.join(self.chckpt_dir, name+'_sac')
        self.activation = get_activation(activation)
        self.n_actions = conv_arch['out_num_actions']
        self.reparam_noise = 1e-6
        
        self.feature_ex_block = create_feature_conv(conv_arch, activation)
        self.linear_block = nn.Sequential(
            nn.Linear(1152, 512),
            self.activation, 
            # nn.Dropout(0.5),
            nn.Linear(512, 128),
            self.activation, 
            # nn.Dropout(0.3),
            nn.Linear(128,64),
            self.activation,
            nn.Linear(64,5)
        )
        self.softmax = nn.Softmax(dim=1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch. device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.to(self.device) 
        init_weights(self)      

    def forward(self, state):
        features = self.feature_ex_block(state)
        prob = self.linear_block(features)
        prob = self.softmax(prob)
        
        # Clamp the variation in this range to increase stability 
        # sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)        

        return prob
    
    def sample_dist(self, state):
        
        # NOTE: we could also update so it only add noise to 0.0. samples

        action_probs = self.forward(state)
        action_distribution = Categorical(action_probs)
        sampled_action = action_distribution.sample()
        # log_probs = action_distribution.logits           

        noise = (action_probs==0).float() * 1e-8
        log_probs = torch.log(action_probs + noise)
        
        return sampled_action, action_probs, log_probs
        
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoit(self): 
        self.load_state_dict(torch.load(self.checkpoint_file))

        

############ Some network utilities for the neural net

def init_weights(network):
    '''Implement Xavier weight initialization'''
    gain = torch.nn.init.calculate_gain(nonlinearity='relu')
    for key in network.state_dict():
        if 'weight' in key:
            # torch.nn.init.xavier_normal_(network.state_dict()[key], gain=nn.init.calculate_gain('relu'))
            # torch.nn.init.kaiming_uniform_(network.state_dict()[key],a= gain, mode='fan_out', nonlinearity='relu')    
            # torch.nn.init.kaiming_normal_(network.state_dict()[key], a=gain, mode='fan_in', nonlinearity='relu')   
            torch.nn.init.xavier_uniform_(network.state_dict()[key], gain=nn.init.calculate_gain('relu'))
            
            # pass
        # if 'bias' in key:
            # torch.nn.init.constant_(network.state_dict()[key], 0.1)
        
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        print("invalid activation function!")
        return None