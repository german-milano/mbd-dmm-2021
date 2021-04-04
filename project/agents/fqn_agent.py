import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax
from tqdm.notebook import tqdm
from utils.memories import GenericMemory
from utils.metrics import ris_behavior_policy, ris_estimator
from utils.early_stopping import EarlyStopper

from agents.generic_agent import GenericAgent


class FQNAgent(GenericAgent):
    
    def __init__(self, gym_env, model, obs_processing_func, batch_size, learning_rate, gamma, dataset, trajectories, n_actions, horizon):
        # Calling parent's constructor
        super(FQNAgent, self).__init__(gym_env, model, obs_processing_func, batch_size, learning_rate, gamma)
        
        # Dataset memory
        self.dataset = GenericMemory(data=dataset)

        # Trajectories' memory
        self.trajectories = GenericMemory(data=trajectories)
        
        # Number of actions for RIS calculation
        self.n_actions = n_actions
        
        # Horizon for RIS calculation
        self.horizon = horizon


    def select_action(self, state, epsilon=0, train=False):
        '''
        Function to select an action according to its current state

                Parameters:
                        state (tuple): environment state
                        current_steps (int): number of steps taken
                        train (boolean): var to control if current run is for training or not  
                        
                Returns:
                        action (tuple): action to be taken on the environment
        '''

        # Get action
        action = super(FQNAgent, self).select_action(state, epsilon, train)
            
        return action
    

    def test_run(self, num_episodes):
        '''
        Function to run tests against an environment

                Parameters:
                        num_episodes (int): number of episodes for the tests
                        
                Returns:
                        rewards (list): accumulated rewards per test episode
        '''

        # Rewards traceability
        rewards = []

        for _ in range(num_episodes):
            done = False
            
            # Observe initial state
            state = self.state_processing_function(self.env.reset()).to(self.device)
            
            current_episode_reward = 0.0

            while not done:

                # Select action using a softmax-based policy 
                action_vec = self.policy_net(state).view(-1,1).squeeze().detach()

                # Since SciPy's softmax has some issues when used alongside np.random.choice (i.e. sum of probabilities isn't exactly 1),
                # it has to be implemented from scratch
                try:
                    action_vec = np.exp(action_vec, dtype='float64') / np.sum(np.exp(action_vec, dtype='float64').numpy(), axis=0)
                    
                    # Select action according to distribution
                    action = np.random.choice(range(self.env.action_space.n), 1, p=action_vec)
                    action = action.item()
                except:
                    # Something went wrong
                    print(f"Action vec: {action_vec}")
                    break

                # Select action according to distribution
                #action = np.random.choice(range(self.env.action_space.n), 1, p=action_vec)
                #action = action.item()

                # Execute action, observe results and process them
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                current_episode_reward += reward

                # Update state
                state = next_state

                if done: 
                    break

            rewards.append(current_episode_reward)

        return rewards


    def train_from_dataset(self, number_episodes, is_test=False, test_run_trials=100, early_stopping=False, es_patience=100): 
        '''
        Function to train a NFQ agent using a pre-generated dataset

                Parameters:
                        number_episodes (int): durantion (in episodes) for the training
                        is_test (boolean): ctrl var for test_run execution
                        test_run_trials (int): number of episodes required to execute test_run
                        early_stopping (boolean): ctrl var for early stopping usage
                        es_patience (int): patience (in episodes) for early stopping
                        
                Returns:
                        episode_rewards (list): accumulated rewards per episode
                        episode_ris (list): RIS value per episode
                        episode_steps (list): number of steps taken per episode
        '''

        episode_rewards = []                      # Rewards traceability
        episode_steps = np.zeros(number_episodes) # Steps traceability
        episode_ris = []

        # Early stopper for training
        early_stopper = None

        if early_stopping:
            if is_test:
                # Use test_run rewards
                early_stopper = EarlyStopper(es_patience, (-1)*float('inf'), 'reward', es_type='best')
            else:
                # Use RIS
                early_stopper = EarlyStopper(es_patience, (-1)*float('inf'), 'RIS', es_type='best')

        # Calculate pi_D once at the beginning
        behavior_policy = ris_behavior_policy(self.trajectories.memory, self.n_actions, len(self.trajectories), self.horizon)

        for ep in tqdm(range(number_episodes), unit=' episodes'):

            # Calculate episode RIS
            ris = ris_estimator(self.trajectories.memory, self.device, self.policy_net, behavior_policy, self.n_actions, self.horizon)

            # Update list with new value
            episode_ris.append(ris)

            for _ in range(int(len(self.dataset)/self.batch_size)):
                # Update model
                self.update_weights_from_dataset()

            # Report on the traning rewards every 100 episodes
            if ep % 100 == 0:
                if is_test:
                    # Every 100 episodes, perform test_run_trials against test env
                    rewards = self.test_run(test_run_trials)
                    # Update list with new value
                    episode_rewards.append(rewards)
                    # Current report
                    print(f"Episode {ep} - Avg. Reward and RIS over the last 100 episodes: \t{np.mean(episode_rewards[-100:])} | {np.mean(episode_ris[-100:])}")
                else:
                    # Current report
                    print(f"Episode {ep} - Avg. RIS over the last 100 episodes: \t{np.mean(episode_ris[-100:])}")

            if early_stopping:
                # Check if it's necessary to keep going
                keep_going = True
                if is_test:
                    keep_going = early_stopper.early_stop(np.mean(episode_rewards[-test_run_trials:]), ep, self.policy_net)
                else:
                    keep_going = early_stopper.early_stop(np.mean(episode_ris[-100:]), ep, self.policy_net)

                if not keep_going:
                    # All out of patience
                    early_stopper.report(ep)

                    if early_stopper.es_type == 'best':
                        # Restore best model
                        self.policy_net = early_stopper.best_model
                        print(f'Best model restoration complete!')
                    else:
                        # No other options at the time
                        pass
                    break
        
        # Current report
        if is_test:
            print(f"Episode {ep} - Avg. Reward and RIS over the last 100 episodes: \t{np.mean(episode_rewards[-100:])} | {np.mean(episode_ris[-100:])}")
        else:
            print(f"Episode {ep} - Avg. RIS over the last 100 episodes: \t{np.mean(episode_ris[-100:])}")

        return episode_rewards, episode_steps, episode_ris


    def update_weights_from_dataset(self):
        '''
        Function to update ANN weights using gradient descent

                Parameters:
                        None
                        
                Returns:
                        None
        '''

        if len(self.dataset) > self.batch_size:
            # Reset gradients
            self.optimizer.zero_grad() 

            # Get mini-batch sample from memory
            batch = self.dataset.sample(self.batch_size)
            states, actions, rewards, dones, next_states = list(zip(*batch))

            # Send tensors to appropriate device
            states = (torch.stack(states)).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            dones = torch.tensor(dones).float().to(self.device)
            next_states = (torch.stack(next_states)).to(self.device)

            # Get Q value for the mini-batch according to policy net
            q_actual = self.policy_net(states).gather(dim=1, index=actions.view(-1,1)).squeeze()
  
            # Get max a' Q value for the next states of the mini-batch
            max_q_next_state = self.policy_net(next_states)
            max_q_next_state = max_q_next_state.max(dim=1)[0]*(1 - dones)
            max_q_next_state = max_q_next_state.detach()

            # Compute DQN's target value
            target = rewards + self.gamma * max_q_next_state
        
            # Compute cost and update weights
            # Should be (predictions, targets)
            self.loss_function(q_actual, target).backward()
            self.optimizer.step()
