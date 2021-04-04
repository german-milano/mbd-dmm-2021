import base64
import glob
import io

import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from IPython.display import HTML
from pyvirtualdisplay import Display


def results_normalizing(number_episodes, rewards, steps):
    '''
    Function that "normalizes" results in order to smooth plots

            Parameters:
                    number_episodes (int): number of episodes used in the training
                    rewards (list): accumulated rewards per episode
                    steps (list): accumulated steps per episode
                    
            Returns:
                    avg_rewards (list): "smoothed" rewards
                    avg_steps (list): "smoothed" steps
                    episode_ticks (list): bin used for smoothing
    '''

    # Average results every N timesteps for better visualization.
    average_range = 100
    episode_ticks = int(number_episodes / average_range)

    avg_rewards = np.array(rewards).reshape((episode_ticks, average_range))
    avg_rewards = np.mean(avg_rewards, axis=1)

    avg_steps = np.array(steps).reshape((episode_ticks, average_range))
    avg_steps = np.mean(avg_steps, axis=1)

    return avg_rewards, avg_steps, episode_ticks


# Función que muestra los resultados de las corridas de un algoritmo
def plot_results(number_episodes, rewards, steps):
    '''
    Function to plot results ("smoothed" beforehand)

            Parameters:
                    number_episodes (int): number of episodes used in the training
                    rewards (list): rewards per episode
                    steps (list): steps per episode
                    
            Returns:
                    None
    '''

    # Normalizing
    avg_rewards, avg_steps, episode_ticks = results_normalizing(number_episodes, rewards, steps)

    # Plot rewards
    plt.plot(range(episode_ticks), avg_rewards)
    plt.title("Episode Accumulated Reward")
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.show()

    # Plot steps per episode
    plt.plot(range(episode_ticks), avg_steps)
    plt.title("Steps needed per episode")
    plt.xlabel("Episode Number")
    plt.ylabel("Number Steps")
    plt.show()


# Vars to display video
display = Display(visible=0, size=(1400, 900))
display.start()

def wrap_env(env):
    '''
    Function to wrap the environment and save the visualization as a video file

            Parameters:
                    env (gym.Env): selected environment
                    
            Returns:
                    env (Monitor): wrapped environment
    '''

    env = Monitor(env, './video', force=True)
    return env


def show_video():
    '''
    Function to enable video recording of gym environment and displaying it

            Parameters:
                    None
                    
            Returns:
                    None
    '''

    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
