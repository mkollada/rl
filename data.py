import numpy as np

from torch.utils.data import Dataset


class CartpoleDataset( Dataset ):
    def __init__( self, X, y ):
        self.X = X
        self.y = y
        
    def __len__( self ):
        return self.y.shape[ 0 ]
    
    def __getitem__( self, idx ):
        X_item = self.X[ idx ]
        y_item = self.y[ idx ]
        
        sample = ( X_item, y_item )
        
        return sample
    
def prepare_cartpole_data( data ):
    
    X = data[ 'obs' ] #np.concatenate( [data[ 'actions' ], data[ 'obs' ]], axis=1 )
    y = data[ 'rewards' ]
    
    
    return X, y