import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from data.dataset import Dataset
from model.model import CBOWModel
from training.optimizer import SGD

class Trainer:
    def __init__(self, model: CBOWModel, optimizer: SGD, dataset: Dataset) -> None:
        """
        Initializes the trainer.

        Args:
            model (CBOWModel): The CBOW model to train.
            optimizer (SGD): The optimizer to use for parameter updates.
            dataset (Dataset): The dataset containing context-target pairs.
        """
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.loss_history = []  # To store loss values for each epoch

    def train(self, num_epochs: int, batch_size: int) -> None:
        """
        Trains the CBOW model.

        Args:
            num_epochs (int): Number of epochs to train.
            batch_size (int): Size of each mini-batch.
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Generate mini-batches
            batches = self.dataset.get_batches(batch_size)

            # Use tqdm to show a progress bar for each epoch
            with tqdm(total=len(batches), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for context_indices, target_indices in batches:
                    # Forward pass
                    probs = self.model(context_indices)

                    # Compute loss
                    loss = self.model.compute_loss(context_indices, target_indices)
                    epoch_loss += loss

                    # Backward pass
                    self.model.backward_loss()

                    # Update parameters
                    self.optimizer.step()

                    # Reset gradients
                    self.optimizer.zero_grad()

                    # Update the progress bar
                    pbar.update(1)

            # Compute average loss for this epoch
            avg_loss = epoch_loss / len(batches)
            self.loss_history.append(avg_loss)

            # Print average loss for this epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def plot_loss(self) -> None:
        """
        Plots the training loss over epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', linestyle='-', color='b')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        fig = plt.gcf()
        fig.savefig('training_loss.pdf')
        plt.show()