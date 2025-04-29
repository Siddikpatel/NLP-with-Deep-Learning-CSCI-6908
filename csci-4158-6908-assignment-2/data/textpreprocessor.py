import re
from typing import List

class TextPreprocessor:
    @staticmethod
    def preprocess(text: str) -> List[str]:
        """
        Preprocesses the text by tokenizing, lowercasing, and removing punctuation.

        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of preprocessed tokens.
        """
        # Lowercase the text
        text = text.lower()

        # Remove punctuation and split into tokens
        tokens = re.findall(r'\b\w+\b', text)

        return tokens