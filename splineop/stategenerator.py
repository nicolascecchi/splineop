import numpy as np 


class StateGenerator():
    def __init__(self, signal):
        self.signal = signal 
    def fit(self, n_states):
        self.n_states = n_states
        self.states = None
    def get_states(self):
        if self.states is None:
            raise ValueError("States have not been generated yet.")
        return self.states

class AmplitudeGenerator(StateGenerator):
    def __init__(self, signal):
        super().__init__(signal)

    def fit(self,n_states,pct,local):
        m = len(self.signal)
        states = np.zeros((m+1 , n_states))
        
        # Compute the inverval around the points
        max_signal = np.max(self.signal)
        min_signal = np.min(self.signal)
        interval = np.abs(max_signal - min_signal)
        delta = interval * pct
        if local:
            for i in range(m):
                start = self.signal[i] - delta / 2
                end = self.signal[i] + delta / 2
                states[i] = np.linspace(start[0], end[0], n_states, True)
            states[-1] = states[-2]
        else:
            for i in range(m):
                states[i] = np.linspace(min_signal, max_signal, n_states, True)
            states[-1] = states[-2]
        self.states = np.expand_dims(states, 2) 

        
    
class RandomGenerator(StateGenerator):
    def __init__(self, signal):
        super().__init__(signal)

    def fit(self, n_states,sd="hall"):
        
        # Implement random state generation logic
        try:
            signal_length, signal_dims = self.signal.shape
        except:
            signal_length, signal_dims = self.signal.shape[0], 1
        self.signal = self.signal.reshape(signal_length, signal_dims)
        states_shape = (signal_length, n_states, signal_dims)
        states = np.zeros(shape=states_shape)
        
        #states[-1] = self.signal[-1]
        if sd=="hall":
            from splineop.sputils import sd_hall_diff
            stddevarray = sd_hall_diff(self.signal,var=False)
        else:
            stddevarray = np.ones(signal_dims) * sd
        
        for dim in range(signal_dims):
            noise = np.random.normal(0, stddevarray[dim], (signal_length,n_states))
            states[:,:,dim] = self.signal[:,dim,None] + noise
        self.states =  np.concat((states, states[-1,:,:][None]))

class NeighborsGenerator(StateGenerator):
    def __init__(self, signal):
        super().__init__(signal)

    def fit(self, n_states):
        if n_states % 2 == 0:
            raise ValueError("n_states must be an odd number for neighborhood generation.")
        if n_states < 3:
            raise ValueError("n_states must be at least 3 for neighborhood generation.")
        # Implement neighborhood state generation logic

        # Points to left/right of the current point
        each_side = (n_states - 1) // 2            
        
        # Make signal a 2D array if it is not already
        # This is needed for the states 
        try:
            signal_length, signal_dims = self.signal.shape
        except:
            signal_length, signal_dims = self.signal.shape[0], 1
            self.signal = self.signal.reshape(signal_length, signal_dims)
        
        states_shape = (signal_length + 1, n_states, signal_dims)
        states = np.zeros(shape=states_shape)
        # Fill "middle" states looking backwards and forwards
        for i in range(each_side, signal_length - each_side):
            states[i] = self.signal[i - each_side : i + each_side + 1]
        # Fill first and last states by repeating the available points
        for i in range(0, each_side):
            states[i] = self.signal[i:n_states+i]
            states[signal_length - each_side + i] = self.signal[-n_states:]
        states[-1] = self.signal[-n_states:]
        self.states = states
