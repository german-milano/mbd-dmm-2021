import pickle

import torch


def process_state(obs):
    return torch.from_numpy(obs).float()


def load_dataset(filename):
    '''
    Function to load a list of samples from Pickle file

            Parameters:
                    filename (str): path to pickle file
                    
            Returns:
                    dataset (list): list of samples as transitions
    '''
    
    # Dataset
    dataset = None
    
    # Open file handler and load data
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)

    return dataset


def save_dataset(filename, dataset):
    '''
    Function to save a list of samples into a Pickle file

            Parameters:
                    dataset (list): list of samples as transitions
                    filename (str): path to pickle file
                    
            Returns:
                    None
                    
    '''
    # Open file handler and dump data
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

