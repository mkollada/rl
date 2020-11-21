import numpy as np
from copy import deepcopy
from torch import Tensor

class Agent:
    '''
    Abstract agent class
    '''
    def __init__( self, action_space, observation_space ):
        
        self.action_space = action_space
        self.obervation_space = observation_space
    
    
    def act():
        raise NotImplementedError
    
class RandomAgent( Agent ):
    '''
    Randomly selects between moving right and left regardless
    '''
    def __init__( self, action_space, obervation_space ):
        super().__init__( action_space, obervation_space )
        
    def act( self, observation ):
        return self.action_space.sample()
    
    
class DeepQAgent( Agent ):
    '''
    Uses model to decide which action to take
    '''
    def __init__( self, action_space, obervation_space, model, discount_rate=0.95, rand_prob=0,
                device=None ):
        super().__init__( action_space, obervation_space )
    
        self.target_model = deepcopy( model ).to( device )
        self.prediction_model = deepcopy( model ).to( device )
        self.device = device
        
        self.discount_rate = discount_rate
        # set as attribute
        self.rand_prob = rand_prob
    
    def get_Q_vals( self, x, model_type ):
        '''
        Returns Q values for all actions in the agents action space for 
        target or prediction model
        
        Args:
            x (np.ndarray): np array with action first then observation
        '''
        if type( x ) != Tensor:
            x = Tensor( x )
            
        x = x.to( self.device )
        
        q_vals = None
        if model_type == 'prediction':
            q_vals = self.prediction_model( x )
        elif model_type == 'target':
            q_vals = self.target_model( x )
        else:
            print( model_type )
            raise ValueError( 'Model type must be either "prediction" or "target"' )

        return q_vals 

    
    def update_target_model( self ):
        '''
        Updates target model by copying from prediction model
        '''
        self.target_model = deepcopy( self.prediction_model )
    
    def act( self, observation ):
        
        observation = Tensor( observation )
        observation = observation.to( self.device )
        
        q_vals = self.get_Q_vals( observation, 'prediction' )
            
        if np.random.rand() < self.rand_prob:
            return self.action_space.sample()
        else:
            return q_vals.argmax().detach().cpu().numpy()
        