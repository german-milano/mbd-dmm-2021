import base64
import glob
import io

import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
from IPython import display as ipythondisplay
from IPython.display import HTML
from pyvirtualdisplay import Display


# Función que "normaliza" los resultados para suavizar las gráficas que se presentan
def results_normalizing(number_episodes, rewards, steps):
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

    
# Función que grafica la comparación de los algoritmos Sarsa y Q-learning
def plot_comparison(number_episodes, data):
    # Tikz
    episode_ticks = int(number_episodes / 100)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,15))
    fig.subplots_adjust(hspace=.3)
    fig.suptitle('DQN vs. Double DQN', 
                 fontsize='x-large', 
                 fontweight='demibold')

    # Accumulated reward
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Accumulated Reward")
    ax1.grid(color='black', alpha= .2, linestyle='--', axis='both', linewidth=1)
    ax1.plot(range(episode_ticks), data['dqn']['rewards'], label=r"DQN")
    ax1.plot(range(episode_ticks), data['ddqn']['rewards'], label=r"D-DQN")

    # Number of iterations
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Steps Needed per Episode")
    ax2.grid(color='black', alpha= .2, linestyle='--', axis='both', linewidth=1)
    ax2.plot(range(episode_ticks), data['dqn']['steps'], label=r"DQN")
    ax2.plot(range(episode_ticks), data['ddqn']['steps'], label=r"D-DQN")

    handles, labels = ax2.get_legend_handles_labels()
    
    # Using ';' to avoid unnecessary prints
    fig.legend(handles, labels, loc='upper right', fontsize='large');


#display = Display(visible=0, size=(1400, 900))
#display.start()

def wrap_env(env):
    """
    Wrapper del ambiente donde definimos un Monitor que guarda la visualizacion como un archivo de video.
    """

    env = Monitor(env, './video', force=True)
    return env


def show_video():
    """
    Utility function to enable video recording of gym environment and displaying it
    To enable video, just do "env = wrap_env(env)""
    """
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
