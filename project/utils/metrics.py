import numpy as np
import torch
from scipy.special import softmax


def ris_weights_and_return(trajectories, device, policy_net, paths_number, behavior_policy, n):
    '''
    Function to calculate the weights and returns for the RIS equation

            Parameters:
                    trajectories (list): dataset of trajectories
                    device (torch.device): device where to run ops in
                    policy_net (torch.nn.Model): artificial neural network
                    path_number (int): number of complete paths in trajectories
                    behavior_policy (dict): estimated policy from the dataset
                    n (int): range of the trajectory
                    
            Returns:
                    ws (list): weights from RIS equation
                    gs (list): returns from RIS equation
    '''
    
    # Initialize weights and returns lists
    ws = np.ones(paths_number)
    gs = np.zeros(paths_number)

    # Main loop
    for x, path in enumerate(trajectories):
        long_ob = []

        states, actions, rewards, _, _ = list(zip(*path))

        gs[x] = sum(rewards)

        for state, action in zip(states, actions):
            state = tuple(state.numpy())

            long_ob.append(state)
            
            state = (torch.tensor(state)).to(device)
            
            # Calculate state-actions valures for both policies
            p = policy_net(state).view(-1,1).squeeze().detach() 
            p = softmax(p)
            p = p[action].numpy()
            q = behavior_policy[tuple(long_ob)][action]

            # Calculate weight for current batch
            ws[x] *= p / q
    
            # Append action
            long_ob.append(action)
    
            # Check horizon
            if len(long_ob) >= n * 2:
                long_ob = long_ob[2:]
        
    return ws, gs


def ris_behavior_policy(trajectories, n_actions, paths_number, n):
    '''
    Function to estimate the behaviour policy from the available data

            Parameters:
                    trajectories (list): dataset of trajectories
                    n_actions (int): number of possible actions
                    paths_number (int): number of complete paths in trajectories
                    n (int): range of the trajectory
                    
            Returns:
                    probs (dict): calculated probabilities
    '''

    # Generate and initialize counters for behaviour policy
    sa_counts = {}
    sa_counts[tuple([])] = paths_number
    s_counts = {}
    s_counts[tuple([])] = paths_number
    
    # Dict for BP probs
    probs = {}
    probs[tuple([])] = paths_number

    # Main loop
    for _, path in enumerate(trajectories):
        # Trajectory's states and actions list
        long_ob = []

        # Use states and actions from path
        states, actions, _, _, _ = list(zip(*path))

        # Batch's states and actions inner loop 
        for state, action in zip(states, actions):

            # Convert state tensors to arrays
            state = tuple(state.numpy())
            
            # Append state
            long_ob.append(state)

            # Check if complete sequence was already seen
            # Otherwise, append it to dict
            if tuple(long_ob) not in s_counts:      
                s_counts[tuple(long_ob)] = 0.0
                sa_counts[tuple(long_ob)] = np.zeros(n_actions)
        
            # Update values
            s_counts[tuple(long_ob)] += 1
            sa_counts[tuple(long_ob)][action] += 1
            long_ob.append(action)

            # Check horizon
            if len(long_ob) >= n * 2:
                long_ob = long_ob[2:]

    # Calculate trajectories' probabilities
    for state in s_counts:
        probs[state] = sa_counts[state] / s_counts[state]

    return probs


def ris_estimator(trajectories, device, policy_net, behavior_policy, n_actions, horizon, n=None, weighted=False):
    '''
    Function to calculate RIS estimator

            Parameters:
                    trajectories (list): dataset of trajectories
                    device (torch.device): device where to run ops in
                    policy_net (torch.nn.Model): artificial neural network
                    behavior_policy (dict): estimated policy from the dataset
                    n_actions (int): number of possible actions
                    horizon (int): number of elements for a complete trajectory
                    n (int): range of the trajectory
                    weighted (boolean): ctrl var for weighted RIS calculation
                    
            Returns:
                    ris (float): RIS estimator value
    '''

    n_val = horizon
    if n is not None:
        n_val = n

    # Number of trajectories to use
    paths_number = len(trajectories)

    # Calculate trajectories' weights and returns
    ws, gs = ris_weights_and_return(trajectories, device, policy_net, paths_number, behavior_policy, n_val)

    # Compute RIS(n) value
    ris = np.dot(gs, ws)
    ris = ris / paths_number

    return ris
