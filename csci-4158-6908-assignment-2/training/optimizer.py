from typing import List

from model.parameter import Parameter


class SGD:
    def __init__(self, parameters: List[Parameter], learning_rate: float = 0.01) -> None:
        """
        Initializes the SGD optimizer.

        Args:
            parameters (list): List of Parameter objects to optimize.
            learning_rate (float): Learning rate for the optimizer.
        """
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self) -> None:
        """
        Performs a single optimization step (parameter update).
        """
        for param in self.parameters:
            if param.grad is not None:
                #TODO: Update the parameter using gradient descent
                #Hint: Remember where the gradients and the parameters are stored.
                param.data -= self.learning_rate * param.grad
                
    def zero_grad(self) -> None:
        """
        Resets the gradients of all parameters to zero.
        """
        for param in self.parameters:
            if param.grad is not None:
                param.zero_grad()