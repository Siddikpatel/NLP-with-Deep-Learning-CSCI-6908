from typing import Dict, List

class Vocabulary:
    def __init__(self):
        """Initializes the vocabulary."""
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0

    def build_vocab(self, tokens: List[str]) -> None:
        """
        Builds the vocabulary from a list of tokens.

        Args:
            tokens (List[str]): List of tokens.
        """
        unique_tokens = set(tokens)
        self.word_to_idx = {word: idx for idx, word in enumerate(unique_tokens)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(unique_tokens)

    def word_to_index(self, word: str) -> int:
        """
        Converts a word to its corresponding index.

        Args:
            word (str): The word to convert.

        Returns:
            int: The index of the word.
        """
        return self.word_to_idx.get(word, -1)  # Return -1 if word is not in vocabulary

    def index_to_word(self, index: int) -> str:
        """
        Converts an index to its corresponding word.

        Args:
            index (int): The index to convert.

        Returns:
            str: The word corresponding to the index.
        """
        return self.idx_to_word.get(index, "<UNK>")  # Return <UNK> if index is not in vocabulary

    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: The size of the vocabulary.
        """
        return self.vocab_size