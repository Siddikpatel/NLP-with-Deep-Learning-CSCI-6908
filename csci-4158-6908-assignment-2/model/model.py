
from model.layers import Embedding, Linear
from model.activations import SoftmaxLayer
from model.loss import NegativeLogLikelihoodLoss
import numpy as np

class CBOWModel:
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int) -> None:
        """
        Initializes the CBOW model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the word embeddings.
            context_size (int): Number of context words (on ONE side of the target word).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        #TODO: Initialize layers with correct dimensions whenever applicable
        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.linear_layer = Linear(2 * context_size * embedding_dim, vocab_size)
        self.softmax_layer = SoftmaxLayer()
        self.loss_fn = NegativeLogLikelihoodLoss()

    def __call__(self, context_indices: np.ndarray) -> np.ndarray:
        """
        Convenience method for calling the forward method.

        Args:
            context_indices (np.ndarray): Array of context word indices
                                         (shape: (batch_size, 2 * context_size)).

        Returns:
            np.ndarray: Predicted probabilities for the target word
                       (shape: (batch_size, vocab_size)).
        """
        return self.forward(context_indices)

    def forward(self, context_indices: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the CBOW model.

        Args:
            context_indices (np.ndarray): Array of context word indices
                                         (shape: (batch_size, 2 * context_size)).

        Returns:
            np.ndarray: Predicted probabilities for the target word
                       (shape: (batch_size, vocab_size)).
        """
        #TODO: Lookup embeddings for context words from the embedding layer.
        embeddings = self.embedding_layer(context_indices) # Shape: (batch_size, 2 * context_size, embedding_dim)

        #TODO: Concatenate embeddings for context words. You need to go from shape (batch_size, 2 * context_size, embedding_dim)
        # to shape (batch_size, 2 * context_size * embedding_dim).
        batch_size = embeddings.shape[0]
        embeddings_reshaped = embeddings.reshape(batch_size, -1)

        #TODO: Feed the `embeddings_reshaped` through the linear layer.
        logits = self.linear_layer(embeddings_reshaped) # Shape: (batch_size, vocab_size)

        #TODO: Convert logits to probabilities using the softmax layer.
        probs = self.softmax_layer(logits) # Shape: (batch_size, vocab_size)

        return probs

    def backward(self, grad_output: np.ndarray) -> None:
        """
        Performs the backward pass of the CBOW model.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output probabilities
                                     (shape: (batch_size, vocab_size)).
        """
        #TODO: Backpropagate through softmax layer using the backward() method.
        #Hint: You need to take as input the gradient of the loss with respect to the output probabilities.
        grad_logits = self.softmax_layer.backward(grad_output) # Shape: (batch_size, vocab_size)

        #TODO: Backpropagate through the linear layer using the backward() method.
        grad_embeddings_reshaped = self.linear_layer.backward(grad_logits)  # Shape: (batch_size, 2 * context_size * embedding_dim)

        # Reshape gradients to match the original embedding shape:
        # Go from (batch_size, 2 * context_size * embedding_dim) to (batch_size, 2 * context_size, embedding_dim)
        batch_size = grad_embeddings_reshaped.shape[0]
        grad_embeddings = grad_embeddings_reshaped.reshape(batch_size, 2 * self.context_size, self.embedding_dim)

        # Backpropagate through embedding layer
        self.embedding_layer.backward(grad_embeddings)

    def compute_loss(self, context_indices: np.ndarray, target_indices: np.ndarray) -> float:
        """
        Computes the loss for a batch of context-target pairs.

        Args:
            context_indices (np.ndarray): Array of context word indices
                                         (shape: (batch_size, 2 * context_size)).
            target_indices (np.ndarray): Array of target word indices
                                        (shape: (batch_size,)).

        Returns:
            float: The computed loss.
        """
        #TODO: Call the class's forward method with the context indices to get the predicted probabilities.
        probs = self.forward(context_indices)

        #TODO: Compute loss by calling the loss function forward method.
        loss = self.loss_fn(probs, target_indices)
        return loss

    def backward_loss(self) -> None:
        """
        Performs the backward pass starting from the loss function.
        """
        # Backpropagate through loss function
        grad_output = self.loss_fn.backward()  # Shape: (batch_size, vocab_size)

        # Backpropagate through the rest of the model
        self.backward(grad_output)

    def zero_grad(self) -> None:
        """
        Resets the gradients of all parameters to zero.
        """
        self.embedding_layer.embeddings.zero_grad()
        self.linear_layer.weights.zero_grad()