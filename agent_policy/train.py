from tqdm import tqdm
import torch, wandb
from utils.helper_fcts import launch_wandb
from copy import deepcopy


def train(args, env, agent, device, wandb_run_name, log_results_wandb=False, variable_lr=False, save_dir=None, save_dir_extension=None):
    '''Training loop.'''

    num_episodes = args.num_episodes
    tau = args.tau
    
    val_check_period = args.val_check_period
    tgt_update_period = args.target_update_period

    val_nb_episodes = args.val_nb_episodes
    # train_nb_episodes = args.train_nb_episodes
    episode_length = args.episode_length

    tgt_update_period = args.target_update_period

    if log_results_wandb:
        launch_wandb(args, wandb_run_name)
        # This will let us see 
        wandb.watch(agent.policy, log="all")

    print(f'Training Start, running on {device}')

    pbar = tqdm(total=num_episodes)

    step_done = 0 
    val_rew = 0.0
    training_reward  = 0.0
    training_loss = 0.0
    val_run_counter=0
    last_val_reward=0
    new_val_reward=0
    for i_episode in range(num_episodes):
        # print("********** training episode # ", i_episode, "**********")
        

################ Training LOOP ################
        state = env.reset('training').unsqueeze(0)
        state = state.to(device)
        training_reward = 0.0
        train_action_history = [] # keep track of which action gets selected to get a feel for the distribution
        
        for i_step in range(episode_length):  # loop over entire episode
            step_done += 1
            action = agent.select_action(state)
            reward, observation, done = env.step(action.item())
            train_action_history.append([i_step, action.item()])

            training_reward += reward

            next_state = observation.unsqueeze(0).to(device).clone().detach()
            reward = torch.tensor(reward, dtype=torch.float32,device=device).unsqueeze(0)
            done = torch.tensor(done, dtype=torch.int, device=device)
            agent.save_experience(state, action, next_state, reward, done)
            state = next_state
                        
            # Upate target network 
            if step_done % tgt_update_period==0:
                # print("********** updating network **********")
                # Soft update of the target network 
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = deepcopy(agent.target).state_dict()
                policy_net_state_dict = deepcopy(agent.policy).state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*tau \
                        + target_net_state_dict[key]*(1-tau)
                agent.target.load_state_dict(target_net_state_dict)
        
            # log loss at each time step
            training_loss = agent.optimize(step_done)
            if training_loss is not None:
                if log_results_wandb:
                    wandb.log({
                        "training_loss for each time step ": training_loss,
                        "training chosen action for each step ": action.item(),
                        "training reward": training_reward
                    })  
                training_loss = 0.0
                
        ############################
        ## Run Validation Reward ####
        if i_episode % val_check_period == 0 and i_episode>0: 

            print("********** new validation run # ", val_run_counter," **********")
            val_rew = 0.0
            val_action_history = [] # keep track of which action gets selected to get a feel for the distribution

            if val_run_counter > 3 or val_run_counter==0:
                val_run_counter=1
                print("********** FULL validation **********")
                save_name = save_dir + '_full_val_eps' + str(i_episode) + save_dir_extension
                torch.save(agent.policy.state_dict(),save_name)
                # agent_script = torch.jit.script(agent.policy)
                # agent_script.save(save_dir + '_full_val_eps' + str(i_episode) + save_dir_extension) 
                range_val_nb_episodes=100
            else:
                range_val_nb_episodes=val_nb_episodes
            val_run_counter+=1

            for j_episode in range(range_val_nb_episodes):
                
                state = env.reset("validation").unsqueeze(0)
                state = state.to(device)
                
                for j_step in range(episode_length):  # loop over 200 steps per episode
                    action = agent.select_action(state, greedy=True)  # get action for the obs from your trained policy
                    rew, next_obs, done = env.step(action.item())  # take one step in the environment
                    val_action_history.append([j_step, action.item()])
                    val_rew += rew  # track rewards
                    state = next_obs.unsqueeze(0).to(device).clone().detach()
                    
                    if log_results_wandb:
                        wandb.log({
                            "val_chosen action for each step ": action.item()
                        })   

            new_val_reward=val_rew/range_val_nb_episodes
            if log_results_wandb:
                wandb.log({
                    "val_reward avg over all val episodes": new_val_reward
                })    
            print("Validation reward : ", new_val_reward)
            print("Training reward : ", training_reward)

            if new_val_reward>last_val_reward:
                last_val_reward=new_val_reward
                save_name = save_dir + "_best_" + save_dir_extension
                torch.save(agent.policy.state_dict(), save_name)
                # agent_script = torch.jit.script(agent.policy)
                # agent_script.save(save_dir + "_best_" + save_dir_extension) 
                print("best model saved")

        # compute schedule of learning rate 
        if variable_lr:
            agent.scheduler.step(val_rew/range_val_nb_episodes)

        # log training reward at each episode
        if log_results_wandb:               
            wandb.log({
                "training_reward for each episode" : training_reward
            })  

        pbar.update(1)  
    pbar.close()
    print('complete')

    # Return 0 for train off
