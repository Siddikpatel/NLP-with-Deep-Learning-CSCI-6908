import numpy as np
from model.parameter import Parameter


class Linear:
    def __init__(self, in_features: int, out_features: int):
        """Initialize linear layer parameters.
        
        Args:
            in_features (int): Size of input features
            out_features (int): Size of output features
        """
        # We first initialize weights using Xavier/Glorot initialization. This is a fancier than a completely random initialization. To do so, we first calculate the bound for the random values and then generate random values uniformly from this range.
        bound = np.sqrt(6.0 / (in_features + out_features))
        # TODO: Generate a random matrix of weights with shape (in_features, out_features). Hint: Check np.random.uniform method.
        weights = np.random.uniform(-bound, bound, (in_features, out_features))

        # TODO: Create a Parameter object to store the weights
        self.weights = Parameter(weights)
        # Cache input for backward pass
        self.input_cache = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of linear layer.
        
        Args:
            x (np.ndarray): Input tensor of shape (batch_size, in_features)
            
        Returns:
            np.ndarray: Output tensor of shape (batch_size, out_features)
        """
        # Cache input for backward pass. We will need this to compute gradients.
        #TODO: This variable should be assigned the input `x`.
        self.input_cache = x
        #TODO: Return the linear transformation
        out = np.dot(x, self.weights.data)
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of linear layer.
        
        Args:
            grad_output (np.ndarray): Gradient of loss with respect to layer output (dL/dy)
                                    Shape: (batch_size, out_features)
                                    
        Returns:
            np.ndarray: Gradient of loss with respect to layer input
                       Shape: (batch_size, in_features)
        """
        # Accumulate gradient for weights: dL/dW
        # Hint: To calculate the transpose of a numpy array `arr`, use `arr.T`
        # TODO: Compute the gradient of loss with respect to weights
        self.weights.grad += np.dot(self.input_cache.T, grad_output)
        
        # TODO: Compute gradient with respect to input: dL/dX
        grad_input = np.dot(grad_output, self.weights.data.T)
        
        return grad_input
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Call the forward method for this layer."""
        return self.forward(x)
    
    def zero_grad(self) -> None:
        """Resets the gradients of the weights to zero."""
        self.weights.zero_grad()


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        Initializes the embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # TODO: Initialize embeddings as a Parameter object
        bound = np.sqrt(6.0 / (vocab_size + embedding_dim))
        self.embeddings = Parameter(np.random.uniform(-bound, bound, (vocab_size, embedding_dim)))

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the embedding layer.

        Args:
            indices (np.ndarray): Array of word indices (shape: (batch_size, context_size)).

        Returns:
            np.ndarray: Embeddings corresponding to the input indices (shape: (batch_size, context_size, embedding_dim)).
        """
        return self.forward(indices)

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the embedding layer.

        Args:
            indices (np.ndarray): Array of word indices (shape: (batch_size, context_size)).

        Returns:
            np.ndarray: Embeddings corresponding to the input indices (shape: (batch_size, context_size, embedding_dim)).
        """
        self.indices = indices  # Save indices for backpropagation

        #TODO: Return the embeddings corresponding to the input indices
        output = self.embeddings.data[indices]
        return output

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Performs the backward pass of the embedding layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output of this layer
                                     (shape: (batch_size, 2 * context_size, embedding_dim)).
        """

        # Initialize gradient for embeddings if not already done
        if not hasattr(self.embeddings, 'grad'):
            self.embeddings.grad = np.zeros_like(self.embeddings.data)

        # TODO: Accumulate gradients for the embeddings corresponding to the input indices.
        np.add.at(self.embeddings.grad, self.indices, grad_output)

    def get_embeddings(self) -> np.ndarray:
        """
        Returns the current word embeddings.

        Returns:
            np.ndarray: Word embeddings (shape: (vocab_size, embedding_dim)).
        """
        return self.embeddings.data