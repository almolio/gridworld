from tqdm import tqdm 
import torch, wandb
from utils.helper_fcts import launch_wandb
from copy import deepcopy

# https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC.py

def train_sac(args, env, agent, 
              device,
              wandb_run_name, 
              save_dir, 
              log_results_wandb=False, 
              load_checkpoint=False
              ):
    
    num_episodes = args.num_episodes
    val_check_period = args.val_check_period
    val_nb_episodes = args.val_nb_episodes
    episode_length = args.episode_length
    high_score = 0
    # Loggin with wandb
    if log_results_wandb:
        launch_wandb(args, wandb_run_name)
        # This will let us see 
        wandb.watch(agent.actor, log="all")
    print(f'Training Start, running on {device}')

    ## Init some other vars 
    step_done = 0 
    val_rew = 0.0
    
    ## start from checkpoit 
    if load_checkpoint: 
        agent.load_models()
        
    for i_episode in tqdm(range(num_episodes)):
        
        state = env.reset('training')
        state = state.to(device)
        training_reward = 0.0
        free_reward_range = 0

        for i_step in range(episode_length):
            action = agent.select_action(state, greedy=False)
            reward, state_, done = env.step(action.item())
            ### IGNORE REWARD IN THE BEGINNING 
            if i_episode < free_reward_range and reward == -1: 
                reward = 0             
            training_reward += reward 
            
            state_ = state_.to(device)
            reward = torch.tensor(reward, dtype=torch.float32, device=device)
            done = torch.tensor(done, dtype=torch.int, device=device)
            agent.save_experience(state, action, state_, reward, done)

            
            for i in range(1):
                loss_dict = agent.learn()
            
            # Reset for nextstep    
            state = state_
            step_done += 1 
            
        ## LOGGING AND DISPLAYING
        # LOG training reward 
        if loss_dict is not None: 
            training_log = {
                "training reward": training_reward,
                **loss_dict}
            if log_results_wandb:               
                wandb.log(training_log) 
            if i_episode % 10 == 0: 
                print(f' training losses: {training_log}')

        ## If training is high, then double check and save 
        if training_reward > 350 and i_episode > free_reward_range:
            full_val = run_validation(env, agent, device, 
                                     episode_length,
                                    i_episode, val_nb_episodes)
            if full_val > 350: 
                savecheckpoint(agent, save_dir, name=f'highEPS{i_episode}')
                print('saving training high score')
        ############################
        ### RUN VALIDATION REWARD ## 
        if i_episode % val_check_period == 0 and i_episode > 0: 
            val_rew = run_validation(env, agent, device, 
                                     episode_length,
                                    i_episode, val_nb_episodes)
            if log_results_wandb: 
                wandb.log({
                    "val_reward averaged" : val_rew
                    })    
            print(f'Validation reward {val_rew}')    
            ## Save High score 
            if val_rew > high_score: 
                print('Best Score saving')
                savecheckpoint(agent, save_dir, name="best")
                high_score = val_rew    

def run_validation(env, agent, device, episode_length,
                   i_episode, val_nb_episodes):
    print(f'******validation run episode {i_episode}******')
    val_rew = 0
    for j_episode in range(val_nb_episodes):
        state = env.reset('validation')
        # TODO: Do we need to turn the agent to eval mode? 
        state = state.to(device)
        for j_step in range(episode_length):
            with torch.no_grad():
                action = agent.select_action(state, greedy=True)
            reward, state_, _ = env.step(action.item())
            val_rew += reward
            state = state_.to(device)
    val_rew = val_rew / val_nb_episodes
    return val_rew

def savecheckpoint(agent, save_dir, name):
    torch.save(agent.actor.state_dict(), save_dir + f"_actor{name}_.pt")
    torch.save(agent.critic_local.state_dict(), save_dir + f"_criticlocal{name}_.pt")
    torch.save(agent.critic_local_2.state_dict(), save_dir + f"_criticlocal2{name}_.pt")
    torch.save(agent.critic_target.state_dict(), save_dir + f"_critictarget{name}_.pt")
    torch.save(agent.critic_target_2.state_dict(), save_dir + f"_critictarget2{name}_.pt")
