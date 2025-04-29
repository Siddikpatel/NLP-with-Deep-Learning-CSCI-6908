import numpy as np

class NegativeLogLikelihoodLoss:
    def __init__(self):
        """Initializes the loss function."""
        self.cache = None  # To store intermediate values for backpropagation

    def forward(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes the negative log likelihood loss.

        Args:
            probs (np.ndarray): Predicted probabilities from the softmax layer
                               (shape: (batch_size, num_classes)).
            targets (np.ndarray): True target indices (shape: (batch_size,)).

        Returns:
            float: The computed loss.
        """
        #TODO: Get the batch size
        batch_size = probs.shape[0]

        #TODO: Get the predicted probabilities for the target indices.
        #Hint: Check np.arange() and numpy fancy indexing.
        probs_target = probs[np.arange(batch_size), targets]

        #TODO: Compute the negative log likelihood loss and do not forget to divide by the batch size.
        #Hint: np.sum() and np.log() might be hepful. You will use the variable probs_target.
        loss = -np.sum(np.log(probs_target)) / batch_size

        # Save values for backpropagation
        self.cache = (probs, targets)
        return loss
    
    def __call__(self, probs: np.ndarray, targets: np.ndarray) -> float:
        """
        Convenience method for calling the forward method.

        Args:
            probs (np.ndarray): Predicted probabilities from the softmax layer
                               (shape: (batch_size, num_classes)).
            targets (np.ndarray): True target indices (shape: (batch_size,)).

        Returns:
            float: The computed loss.
        """
        return self.forward(probs, targets)

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input probabilities.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input probabilities
                       (shape: (batch_size, vicab_size)).
        """
        probs, targets = self.cache
        batch_size, num_classes = probs.shape

        # Initialize gradient as zeros
        grad_input = np.zeros_like(probs)

        # Set the gradient for the target indices to -1/probs_target
        grad_input[np.arange(batch_size), targets] = -1.0 / probs[np.arange(batch_size), targets]

        # Normalize by batch size
        grad_input /= batch_size
        return grad_input