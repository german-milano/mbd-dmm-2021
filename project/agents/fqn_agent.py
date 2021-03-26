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
        # Llamo al constructor del parent
        super(FQNAgent, self).__init__(gym_env, model, obs_processing_func, batch_size, learning_rate, gamma)
        
        # Asigno el dataset
        self.dataset = GenericMemory(data=dataset)

        # Memory para las trayectorias
        self.trajectories = GenericMemory(data=trajectories)
        
        # Asigno numero de acciones
        self.n_actions = n_actions
        
        # Asigno horizonte de RIS
        self.horizon = horizon


    def select_action(self, state, epsilon=0, train=False):
        ### TODO (gmilano): docstrings

        # Obtengo la acción (solo uso greedy)
        action = super(FQNAgent, self).select_action(state, epsilon, train)
            
        return action
    

    def test_run(self, num_episodes):
        ### TODO (gmilano): docstrings
        ### TODO (gmilano): implementar para diferentes tipos de políticas

        # Trazabilidad de los rewards por episodio
        rewards = []

        for _ in range(num_episodes):
            done = False
            
            # Observar estado inicial como indica el algoritmo
            state = self.state_processing_function(self.env.reset()).to(self.device)
            
            current_episode_reward = 0.0

            while not done:

                # Seleccionar accion usando una política basada en softmax
                action_vec = self.policy_net(state).view(-1,1).squeeze().detach()

                # Es necesario implementar softmax 'de cero' porque np tiene un bug asociado a la precisión y 
                # eso hace que la suma del softmax de SciPy, no sea igual a 1 (y esto da problemas con np.random.choice)
                try:
                    action_vec = np.exp(action_vec, dtype='float64') / np.sum(np.exp(action_vec, dtype='float64').numpy(), axis=0)
                    
                    # Selecciono la acción de acuerdo a la distribución
                    action = np.random.choice(range(self.env.action_space.n), 1, p=action_vec)
                    action = action.item()
                except:
                    print(f"Action vec: {action_vec}")
                    break

                # Selecciono la acción de acuerdo a la distribución
                action = np.random.choice(range(self.env.action_space.n), 1, p=action_vec)
                action = action.item()

                # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.state_processing_function(next_state).to(self.device)

                current_episode_reward += reward

                # Actualizar el estado
                state = next_state

                if done: 
                    break

            rewards.append(current_episode_reward)

        return rewards


    def train_from_dataset(self, number_episodes, is_test=False, test_run_trials=100, early_stopping=False, es_patience=100): 
        ### TODO (gmilano): docstrings

        episode_rewards = []                      # Trazabilidad de los rewards por episodio
        episode_steps = np.zeros(number_episodes) # Trazabilidad de steps por episodio
        episode_ris = []

        # Early stopper para el entrenamiento
        early_stopper = None

        if early_stopping:
            if is_test:
                # Utilizo los rewards de test_run
                early_stopper = EarlyStopper(es_patience, (-1)*float('inf'), 'reward', es_type='best')
            else:
                # Utilizo RIS
                early_stopper = EarlyStopper(es_patience, (-1)*float('inf'), 'RIS', es_type='best')

        # Calculo pi_D una única vez al inicio
        behavior_policy = ris_behavior_policy(self.trajectories.memory, self.n_actions, len(self.trajectories), self.horizon)

        for ep in tqdm(range(number_episodes), unit=' episodes'):

            # Calculamos valor RIS del episodio
            ris = ris_estimator(self.trajectories.memory, self.device, self.policy_net, behavior_policy, self.n_actions, self.horizon)

            # Actualizo ope para el episodio
            episode_ris.append(ris)

            for _ in range(int(len(self.dataset)/self.batch_size)):
                # Actualizar el modelo
                self.update_weights_from_dataset()

            # Report on the traning rewards every 100 episodes
            if ep % 100 == 0:
                if is_test:
                    # Cada 100 episodios, hago 'test_run_trials' contra el ambiente
                    rewards = self.test_run(test_run_trials)
                    # Actualizo
                    episode_rewards.append(rewards)
                    # Current report
                    print(f"Episode {ep} - Avg. Reward and RIS over the last 100 episodes: \t{np.mean(episode_rewards[-100:])} | {np.mean(episode_ris[-100:])}")
                else:
                    # Current report
                    print(f"Episode {ep} - Avg. RIS over the last 100 episodes: \t{np.mean(episode_ris[-100:])}")

            if early_stopping:
                # Verifico si debo finalizar
                keep_going = True
                if is_test:
                    keep_going = early_stopper.early_stop(np.mean(episode_rewards[-test_run_trials:]), ep, self.policy_net)
                else:
                    keep_going = early_stopper.early_stop(np.mean(episode_ris[-100:]), ep, self.policy_net)

                if not keep_going:
                    # Se me terminó la paciencia
                    early_stopper.report(ep)

                    if early_stopper.es_type == 'best':
                        # Restauro el mejor modelo
                        self.policy_net = early_stopper.best_model
                        print(f'Best model restoration complete!')
                    else:
                        # TODO (gmilano): implementar otras opciones
                        pass
                    break
        
        # Current report
        if is_test:
            print(f"Episode {ep} - Avg. Reward and RIS over the last 100 episodes: \t{np.mean(episode_rewards[-100:])} | {np.mean(episode_ris[-100:])}")
        else:
            print(f"Episode {ep} - Avg. RIS over the last 100 episodes: \t{np.mean(episode_ris[-100:])}")

        return episode_rewards, episode_steps, episode_ris


    def update_weights_from_dataset(self):
        ### TODO (gmilano): docstrings

        if len(self.dataset) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad() 

            # Obtener un minibatch de la memoria 
            batch = self.dataset.sample(self.batch_size)
            states, actions, rewards, dones, next_states = list(zip(*batch))

            # Enviar los tensores al dispositivo correspondiente.
            states = (torch.stack(states)).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            dones = torch.tensor(dones).float().to(self.device)
            next_states = (torch.stack(next_states)).to(self.device)

            # Obetener el valor estado-accion (Q) de acuerdo a la policy net para todo elemento (estados) del minibatch.
            q_actual = self.policy_net(states).gather(dim=1, index=actions.view(-1,1)).squeeze()
  
            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este computo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.          
            max_q_next_state = self.policy_net(next_states)
            max_q_next_state = max_q_next_state.max(dim=1)[0]*(1 - dones)
            max_q_next_state = max_q_next_state.detach()

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            target = rewards + self.gamma * max_q_next_state
        
            # Compute el costo y actualice los pesos.
            # En Pytorch la funcion de costo se llaman con (predicciones, objetivos) en ese orden.
            self.loss_function(q_actual, target).backward()
            self.optimizer.step()
