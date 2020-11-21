import numpy as np

def run_episode( env, agent, timesteps=100, verbose=False, render=False ):
    observation = env.reset()
    
    observations = np.zeros( ( timesteps, 4 ) )
    rewards = np.zeros( ( timesteps, 2 ) )
    actions = np.zeros( ( timesteps, 1 ) )
    
    data = {}
    for t in range( timesteps ):
        if render:
            env.render()
        observations[ t ] = observation
        
        # Real action given current observation
        action = agent.act( observation )
        actions[ t ] = action
        
        observation, r, done, info = env.step( action )
        
        reward = 1 - 2*done
        
        rewards[ t ] = reward + agent.discount_rate * agent.get_Q_vals( observation, 'target' ).detach().numpy()
        
        if done == 1:
            if verbose:
                print("Episode finished after {} timesteps".format(t+1))
            break
            

    return observations[ :t ], rewards[ :t ], actions[ :t ], t


def run_n_episodes( env, agent, n_episodes, timesteps=100 ):
    data = {}
    data[ 'obs' ] = np.zeros( ( 0, 4 ) ) 
    data[ 'rewards' ] = np.zeros( ( 0, 2 ) ) 
    data[ 'actions' ] = np.zeros( ( 0, 1 ) )
    data[ 'timesteps' ] = []
    for i in range( n_episodes ):
        data[ i ] = {}
        
        observations, rewards, actions, t = run_episode( env, agent, timesteps=timesteps )
        data[ 'obs' ] = np.concatenate( [ data[ 'obs' ], observations ], axis=0 )
        data[ 'rewards' ] = np.concatenate( [ data[ 'rewards' ], rewards ], axis=0 )
        data[ 'actions' ] = np.concatenate( [ data[ 'actions' ], actions ], axis=0 )
        data[ 'timesteps' ].append( t )
        
    env.close()
    
    return data