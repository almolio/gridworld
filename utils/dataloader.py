from os import walk
import pandas, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple, deque

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class GridDataLoader:
    """This class stores the data for a given variant (0,1,2) and a given data set (train, val, test)"""
   
    def __init__(self, variant, mode, replay_buf_size):
        self.root_folder='../data'
        self.variants=['/variant_0','/variant_1','/variant_2']
        self.data_folder='/episode_data'
        self.file_ranges = {'training': [200,999], 'testing': [0,99], 'validating': [100,199]  }
        self.grid_world = np.zeros([5,5])
        self.variant = variant
        self.mode = mode
        self.file_range = range(self.file_ranges[self.mode][0], self.file_ranges[self.mode][1])
        self.mypath= self.root_folder + self.variants[self.variant] + self.data_folder
        self.filenames = []
        self.distribution = np.zeros([5,5])
        self.load_all_episodes()
        self.compute_distribution()
        self.memory = deque([], maxlen=replay_buf_size)
        
    def push(self, *args):
        '''Append to the buffer'''
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        '''Sample from the buffer'''
        return random.sample(self.memory, batch_size)

    def load_rdn_episode(self):
        """Randomly select and return an episode as an array of arrays with format [timestep, x, y]"""        
        filename = np.random.randint(low=0, high=len(self.filenames))
        df = pandas.read_csv(self.filenames[filename])
        episode = []

        for line in range(df.shape[0]):
            t = df.loc[line][1]
            y = df.loc[line][2]
            x = df.loc[line][3]
            self.distribution[x,y] +=1
            episode.append([t,x,y])
        return episode
    
    def display_distribution(self):
        """Display the item count over all episodes"""
        plt.imshow(self.distribution, cmap='hot', interpolation='nearest')
        with sns.axes_style("white"):
            ax = sns.heatmap(self.distribution, vmax=1.1*np.max(self.distribution), square=True,  cmap="Reds")
            ax.set_title('Item count ' + str(self.variant) + ' ' + self.mode, fontsize=20)
            ax.set_xlabel('x-horizontal', fontsize=16)
            ax.set_ylabel('y-vertical', fontsize=16)
            ax.figure.set_size_inches(10, 10)
            for i in range(5):
                for j in range(5):
                    text = ax.text(j+0.3, i+0.5, int(self.distribution[i, j]), color="k", fontsize=16)
            plt.show()

    def load_all_episodes(self):
        """Load and return the filenames of all episodes from the data set""" 
        names = []
        for (dirpath, dirnames, filenames) in walk(self.mypath):
            for filename in filenames:
                for file_nr in self.file_range :
                    
                    if file_nr < 100:
                        if file_nr < 10:
                            file_nr = str(file_nr)
                            file_nr = '00' + file_nr
                        else:
                            file_nr = str(file_nr)
                            file_nr = '0' + file_nr
                    else:
                        file_nr = str(file_nr)

                    if file_nr in filename:
                        names.append(dirpath + '/' + filename)
        self.filenames = names          
            
    def compute_distribution(self):
        """Compute the item count over all episodes"""
        distribution = np.zeros_like(self.grid_world)
        for (dirpath, dirnames, filenames) in walk(self.mypath):
            for filename in filenames:
                for file_nr in self.file_range :
                    
                    if file_nr < 100:
                        if file_nr < 10:
                            file_nr = str(file_nr)
                            file_nr = '00' + file_nr
                        else:
                            file_nr = str(file_nr)
                            file_nr = '0' + file_nr
                    else:
                        file_nr = str(file_nr)

                    if file_nr in filename:
                        df = pandas.read_csv(dirpath + '/' + filename)

                        for line in range(df.shape[0]):
                            y = df.loc[line][2]
                            x = df.loc[line][3]
                            distribution[x,y] +=1
        self.distribution = distribution
        return