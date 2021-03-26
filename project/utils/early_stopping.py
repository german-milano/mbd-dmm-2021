

class EarlyStopper:
    
    def __init__(self, patience, init_val, metric_name, es_type='best'):
        
        # Tipo de Early Stopping ('best' = restaura el mejor modelo, 'last':  restaura el último modelo)
        self.es_type = es_type 

        # Paciencia
        self.patience = patience

        # Paciencia actual
        self.curr_patience = patience

        # Métrica
        self.metric = metric_name

        # Mejor valor de la métrica
        self.best_val = init_val

        # Mejor modelo
        self.best_model = None

        # Mejor iteracion
        self.best_episode = None


    
    def early_stop(self, curr_val, episode_num, model):
        # TODO (gmilano): docstrings

        keep_going = True

        if self.curr_patience > 0:
            # Si aún tengo paciencia, sigo
            if  self.best_val < curr_val:
                # Encontré un modelo mejor
                self.best_model = model
                
                # Almaceno el episodio
                self.best_episode = episode_num

                # Reinicio paciencia
                self.curr_patience = self.patience

                # Actualizo el mejor valor
                self.best_val = curr_val
            else:
                # Resto paciencia
                self.curr_patience -= 1
        else:
            # Se me terminó la paciencia
            keep_going = False
        
        return keep_going


    def report(self, curr_ep):
        print(f'\nTraining finished by Early Stopping on episode {curr_ep}')
        print(f'Metric used: {self.metric}')
        print(f'Best {self.metric} value: {self.best_val}')
        print(f'Best episode: {self.best_episode}')
