import inspect

class BaseConfig:
    def __init__(self) -> None:
        """Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                BaseConfig.init_member_classes(i_var)

class DQNCfg(BaseConfig):
    class training:
        BATCH_SIZE = 64
        GAMMA = 0.96
        EPS_START = 0.99
        EPS_END = 0.1
        EPS_DECAY = 1e6
        TAU = 1  # fix soft update when set to 1 
        LR = 1e-4
        LR_DECAY = 1e3   
        NUM_EPISODES=2000
        MEMORY_SIZE = 500
        TARGET_UPDATE_PERIOD = 100
        EPISODE_LENGTH = 200
        VAL_CHECK_PERIOD = 100
        VAL_NB_EPISODES = 100
        # TRAIN_NB_EPISODES = 800

    class agent:
        n_actions = 5
        num_didden_layers = 3
        layer_dim = 64
        conv_arch={'stride':(1,1), 
                   'num_channel': 3, 
                   'kernel_size':(3,3), 
                   'out_num_actions' : 5,
                   'padding' : 1,
                   'cnn_layer_depths': (10,20)} # NOTE: code is made for max 2 cnn layers, here define the number of filters in each layer