class TextLoader:
    @staticmethod
    def load_text(file_path: str) -> str:
        """
        Loads text data from a file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: The content of the text file as a string.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text