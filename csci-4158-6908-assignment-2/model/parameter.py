import numpy as np

class Parameter:
    def __init__(self, data):
        """Initialize a parameter with data.
        
        Args:
            data (np.ndarray): Initial parameter values
        """
        self.data = data
        self.grad = np.zeros_like(data)
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)
    
    def __repr__(self):
        """String representation of the parameter."""
        return f"Parameter(data shape={self.data.shape}, grad shape={self.grad.shape})"
    
    @property
    def shape(self):
        """Return the shape of the parameter data."""
        return self.data.shape
    
    def __getitem__(self, key):
        """Enable indexing of the parameter data."""
        return self.data[key]
    
    def __setitem__(self, key, value):
        """Enable setting values through indexing."""
        self.data[key] = value