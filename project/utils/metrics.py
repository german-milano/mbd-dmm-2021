import numpy as np
import torch
from scipy.special import softmax


def ris_weights_and_return(trajectories, device, policy_net, paths_number, behavior_policy, n):
    # TODO (gmilano): docstrings
    # TODO (gmilano): implementar para diferentes políticas
    
    # Inicio pesos en 1 y retorno en 0
    ws = np.ones(paths_number)
    gs = np.zeros(paths_number)

    # Loop principal
    for x, path in enumerate(trajectories):
        long_ob = []

        states, actions, rewards, _, _ = list(zip(*path))

        gs[x] = sum(rewards)

        for state, action in zip(states, actions):
            state = tuple(state.numpy())    # TODO (gmilano): hacerlo fuera de la función

            long_ob.append(state)
            
            state = (torch.tensor(state)).to(device)
            
            # Calculo los valores para los estados-acciones de ambas politicas utiliando net y prob. calculadas
            # TODO (gmilano): sacar p y q fuera de esta función
            p = policy_net(state).view(-1,1).squeeze().detach() 
            p = softmax(p)  # TODO (gmilano): pasar a np.exp(action_vec, dtype='float64') / np.sum(np.exp(action_vec, dtype='float64').numpy(), axis=0)
            p = p[action].numpy()
            q = behavior_policy[tuple(long_ob)][action]

            # Calculamos el peso del batch
            ws[x] *= p / q
    
            #Agregamos la accion
            long_ob.append(action)
    
            #Chequeo Horizonte
            if len(long_ob) >= n * 2:
                long_ob = long_ob[2:]
        
    return ws, gs


def ris_behavior_policy(trajectories, n_actions, paths_number, n):
    # TODO (gmilano): docstrings
    ''' Aproxima la behavior policy segun los datos'''

    # Genero e inicializo Contadores para Behavior Policy
    sa_counts = {}
    sa_counts[tuple([])] = paths_number
    s_counts = {}
    s_counts[tuple([])] = paths_number
    
    # Diccionario de Behavior Policy con probabilidades
    probs = {}
    probs[tuple([])] = paths_number

    #Loop general sobre las trayectorias 
    for _, path in enumerate(trajectories):
        #Lista para estados y acciones de la trayectoria
        long_ob = []

        #Utilizo los estados y acciones del path
        states, actions, _, _, _ = list(zip(*path))

        # Genero loop sobre estados y acciones dentro del batch
        for state, action in zip(states, actions):

            # Pasamos los estados de tensores a narrays
            state = tuple(state.numpy())
            
            # Agregamos el estado
            long_ob.append(state)

            #Chequeo si ya vi la secuencia completa y la agrego en diccionario en caso contrario
            if tuple(long_ob) not in s_counts:      
                s_counts[tuple(long_ob)] = 0.0
                sa_counts[tuple(long_ob)] = np.zeros(n_actions)
        
            #Actualizo valores de trayectoria al momento
            s_counts[tuple(long_ob)] += 1
            sa_counts[tuple(long_ob)][action] += 1
            long_ob.append(action)

            #Chequeo horizonte
            if len(long_ob) >= n * 2:
                long_ob = long_ob[2:]

    #Calculamos las probabilidades de las trayectorias
    for state in s_counts:
        probs[state] = sa_counts[state] / s_counts[state]

    return probs


def ris_estimator(trajectories, device, policy_net, behavior_policy, n_actions, horizon, n=None, weighted=False):
    # TODO (gmilano): docstrings

    n_val = horizon
    if n is not None:
        n_val = n

    # Numero de trayetorias a utilizar
    paths_number = len(trajectories)

    # Calculo los pesos para cada una de las trayectorias y sus rewards (se hace junto para aprovechar el loop)
    ws, gs = ris_weights_and_return(trajectories, device, policy_net, paths_number, behavior_policy, n_val)

    # Caclulamos valor RIS(n)
    ris = np.dot(gs, ws)
    ris = ris / paths_number

    return ris
