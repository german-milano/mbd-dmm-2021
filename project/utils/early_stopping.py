

class EarlyStopper:
    
    def __init__(self, patience, init_val, metric_name, es_type='best'):
        
        # Early stopping strategy ('best' = restores best model, 'last': restores last model)
        self.es_type = es_type 

        # Patience
        self.patience = patience

        # Current patience value
        self.curr_patience = patience

        # Metric to be used
        self.metric = metric_name

        # Best metric's value
        self.best_val = init_val

        # Best model
        self.best_model = None

        # Best episode
        self.best_episode = None


    
    def early_stop(self, curr_val, episode_num, model):
        '''
        Function to stop an agent's training and restore ANN weights

                Parameters:
                        curr_val (float): current metric's value
                        episode_num (int): current episode number of the training
                        model (nn.Module): ANN model
                        
                Returns:
                        keep_going (boolean): ctrl var to indicate if training should continue
        '''

        keep_going = True

        if self.curr_patience > 0:
            # If there's still patience, check values
            if  self.best_val < curr_val:
                # Better model found
                self.best_model = model
                
                # Save episode number
                self.best_episode = episode_num

                # Reset patience
                self.curr_patience = self.patience

                # Update best metric value
                self.best_val = curr_val
            else:
                # Reduce patience by 1
                self.curr_patience -= 1
        else:
            # All out of patience
            keep_going = False
        
        return keep_going


    def report(self, curr_ep):
        '''
        Function to create report when early stopping is triggered

                Parameters:
                        curr_ep (int): current episode number of the training
                        
                Returns:
                        None
        '''

        print(f'\nTraining finished by Early Stopping on episode {curr_ep}')
        print(f'Metric used: {self.metric}')
        print(f'Best {self.metric} value: {self.best_val}')
        print(f'Best episode: {self.best_episode}')
