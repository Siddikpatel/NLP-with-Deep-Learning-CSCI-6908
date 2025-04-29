import numpy as np
from typing import List, Tuple
from data.vocabulary import Vocabulary

class Dataset:
    def __init__(self, tokens: List[str], vocabulary: Vocabulary, context_size: int):
        """
        Initializes the dataset.

        Args:
            tokens (List[str]): List of preprocessed tokens.
            vocabulary (Vocabulary): The vocabulary object.
            context_size (int): Number of context words on each side of the target word.
        """
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.context_size = context_size

    def generate_context_target_pairs(self) -> List[Tuple[List[int], int]]:
        """
        Generates (context, target) pairs from the tokens.

        Returns:
            List[Tuple[List[int], int]]: A list of (context, target) pairs.
        """
        pairs = []
        for i in range(self.context_size, len(self.tokens) - self.context_size):
            target_word = self.tokens[i]
            context_words = (
                self.tokens[i - self.context_size : i] + self.tokens[i + 1 : i + self.context_size + 1]
            )
            target_index = self.vocabulary.word_to_index(target_word)
            context_indices = [self.vocabulary.word_to_index(word) for word in context_words]
            pairs.append((context_indices, target_index))
        return pairs
    
    def generate_validation_pairs(self, validation_tokens: List[str]) -> List[Tuple[List[int], int]]:
        """
        Generates (context, target) pairs for the validation set.

        Args:
            validation_tokens (List[str]): List of preprocessed tokens for the validation set.

        Returns:
            List[Tuple[List[int], int]]: A list of (context, target) pairs.
        """
        pairs = []
        for i in range(self.context_size, len(validation_tokens) - self.context_size):
            target_word = validation_tokens[i]
            context_words = (
                validation_tokens[i - self.context_size : i] + validation_tokens[i + 1 : i + self.context_size + 1]
            )
            target_index = self.vocabulary.word_to_index(target_word)
            context_indices = [self.vocabulary.word_to_index(word) for word in context_words]
            pairs.append((context_indices, target_index))
        return pairs

    def get_batches(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Splits the data into mini-batches.

        Args:
            batch_size (int): Size of each mini-batch.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of (context, target) batches.
        """
        pairs = self.generate_context_target_pairs()
        np.random.shuffle(pairs)  # Shuffle the pairs

        batches = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            context_indices = np.array([item[0] for item in batch])
            target_indices = np.array([item[1] for item in batch])
            batches.append((context_indices, target_indices))
        return batches