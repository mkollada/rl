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
    def __init__( self, action_space, obervation_space, model, discount_rate=0.95, rand_prob=0 ):
        super().__init__( action_space, obervation_space )
    
        self.target_model = deepcopy( model )
        self.prediction_model = deepcopy( model )
        self.discount_rate = discount_rate
        # set as attribute
        self.rand_prob = rand_prob
        
#     def get_Q( self, x, model ):
#         '''
#         Returns Q Value of either target or prediction model for any x   
        
#         Args:
#             x (np.ndarray): np array with action first then observation
#             model (nn.Module): either target or prediction model
#         '''
#         return model( Tensor( x ) )
    
    def get_Q_vals( self, x, model ):
        '''
        Returns Q values for all actions in the agents action space for 
        target or prediction model
        
        Args:
            x (np.ndarray): np array with action first then observation
        '''
        
#         q_vals = {}
        
#         for i in range( self.action_space.n ):
                
#             x = np.concatenate( [ np.array( [i] ), observation ], axis=-1 )
#             q_vals[ i ] = self.get_Q( x, model )
            
#         return q_vals

        return model( Tensor( x ) ) 
    
    def get_max_Q_val( self, observation, model ):
        '''
        Returns the max Q value and index of the action that generated it 
        from either the prediction or target model
        '''
        
        q_vals = self.get_Q_vals( observation, model )

                
        return q_vals.argmax(), q_vals.max()

    
    def update_target_model( self ):
        '''
        Updates target model by copying from prediction model
        '''
        self.target_model = deepcopy( self.prediction_model )
    
    def act( self, observation ):
        
        best_action, max_q = self.get_max_Q_val( observation, self.prediction_model )
            
        if np.random.rand() < self.rand_prob:
            return self.action_space.sample()
        else:
            return best_action.detach().numpy()
        
    def get_max_target_Q_val( self, observation ):
        '''
        Returns the max Q value of the target model
        '''
        idx_max, max_q = self.get_max_Q_val( observation, self.target_model )
        
        return max_q