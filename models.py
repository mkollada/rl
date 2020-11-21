import torch
import torch.nn as nn

class LSTM( nn.Module ):
    def __init__( self, input_dim, hidden_dim, lstm_layer_dim, dropout, 
                 device=None ):
        '''
        Object that creates encoders with an LSTM architecture
        
        Args:
            input_dim (int): Input dimension of LSTM
            hidden_dim (int): Number of features in the hidden state
            lstm_layers (int): Number of recurrent layers
            output_dim (int): Output dimesnsion of LSTM
            dropout (float): Dropout rate in all but last recurrent layer
            device (str): Device for model to be stored
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm_layer_dim = lstm_layer_dim
        self.dropout = dropout
        
        self.lstm = nn.LSTM( input_dim, hidden_dim, lstm_layer_dim, 
                            batch_first=True, dropout=dropout )
        
        self.device = device
        
    
    def forward( self, x ):
        '''
        Required Pytorch Function. Defines the forward pass for this encoder
        ''' 
        
        # Initialize hidden state with zeros
        h0 = torch.zeros( self.lstm_layer_dim, 
                         x.size(0), 
                         self.hidden_dim )
        
        if self.device:
            h0 = h0.to( self.device )
        
        # Initialize cell state
        c0 = torch.zeros( self.lstm_layer_dim, 
                         x.size(0), 
                         self.hidden_dim )
        if self.device:
            c0 = c0.to( self.device )
    
        
        out, (hn, cn) = self.lstm( x.float(), ( h0, c0 ) )
        
        return out
    
    
class FullyConnected( nn.Module ):
    '''
    FullyConnected NN
    '''
    def __init__( self, input_dim, hidden_dims, dropout=0.1 ):
        super().__init__()
        
        
        layers = []
        
        layers.append( nn.Linear( input_dim, hidden_dims[ 0 ] ) )
        
        for i in range( len( hidden_dims ) - 1 ):
            layers.append( nn.Dropout( dropout ) )
            layers.append( nn.ReLU() )
            linear = nn.Linear( hidden_dims[ i ],
                                    hidden_dims[ i+1 ] )
            nn.init.kaiming_normal_( linear.weight, mode='fan_in' )
            layers.append( linear  )
            
            
        self.layers = nn.Sequential( *layers )
        
        
        
    def forward( self, x ):
        
        return self.layers( x )
    
    
class StackedLSTM( nn.Module ):
    '''
    Allows LSTMs to be stacked with different hidden dimenions. Args should be 
    lists, the length of which is equal to the number of stacked lstms.
    
    Args:
        input_dim (int): input dimension for first lstm
        hidden_dim_list (list): list of number of hidden dimensions for each lstm
        num_layers_list (list): list of number of layers for each lstm
        dropout_list (list): list of dropout value for each lstm
        
    '''
    def __init__ ( self, input_dim, hidden_dim_list, num_layers_list, dropout_list ):
        super().__init__()
        
        if len( hidden_dim_list ) != len( num_layers_list ):
            raise ValueError( 'Hidden dimension list and num layers list should be same length' )
            
        if len( hidden_dim_list ) != len( dropout_list ):
            raise ValueError( 'Hidden dimension list and num layers list should be same length' )
            
        lstms = []
        self.num_lstms = len( hidden_dim_list )
        
        for i in range( self.num_lstms ):
            lstm = None
            if i == 0:
                lstm = LSTM( input_dim, 
                            hidden_dim_list[ i ], 
                            num_layers_list[ i ],
                            dropout_list[ i ] )
            else:
                lstm = LSTM( hidden_dim_list[ i - 1 ], 
                            hidden_dim_list[ i ], 
                            num_layers_list[ i ],
                            dropout_list[ i ] )
        
            lstms.append( lstm )
            
        self.lstms = nn.Sequential( *lstms )
        
        
    def forward( self, x ):
        '''
        Required pytorch function
        '''
        
        return self.lstms( x )
        
    