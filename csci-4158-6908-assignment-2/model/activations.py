import numpy as np

class SoftmaxLayer:
    def __init__(self):
        """Initializes the softmax layer."""
        self.cache = None  # To store intermediate values for backpropagation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the softmax layer.

        Args:
            x (np.ndarray): Input data (shape: (batch_size, num_classes)).

        Returns:
            np.ndarray: Softmax probabilities (shape: (batch_size, num_classes)).
        """
        #TODO: Subtract max value to avoid overflow.
        #Hint: Use np.max with axis=1 and keepdims=True.
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x_shifted) # This is the numerator
        probs = x_exp / np.sum(x_exp, axis=1, keepdims=True) # This is the denominator. Remember to sum over the correct axis.

        self.cache = probs  # Save probabilities for backpropagation
        return probs
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Convenience method for calling the forward method.

        Args:
            x (np.ndarray): Input data (shape: (batch_size, num_classes)).

        Returns:
            np.ndarray: Softmax probabilities (shape: (batch_size, num_classes)).
        """
        return self.forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the softmax layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of this layer
                                     (shape: (batch_size, num_classes)).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer
                       (shape: (batch_size, num_classes)).
        """
        probs = self.cache

        # Compute gradient of softmax input
        grad_input = probs * (grad_output - np.sum(grad_output * probs, axis=1, keepdims=True))
        return grad_input