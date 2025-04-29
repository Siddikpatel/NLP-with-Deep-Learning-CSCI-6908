from scripts.pg_clean_text import strip_headers
from scripts.utils import read_file
from glob import iglob
import os
from typing import List
import re

def read_and_concat_books(books_repo: str) -> str:
    """
    This function will read the content of all the books in the given path and return a single string with all the content concatenated
    on top of each other.
    Example: 
    - Book1: "Hello World"
    - Book2: "Python is cool"
    - Output: "Hello World\n
              Python is cool"
    """
    # Read all books saved in .txt files
    books = list(iglob(os.path.join(books_repo, "*.txt")))
    books_content = list(map(read_file, books))
    # Remove the GP headers
    cleaned_contents = list(map(lambda x: strip_headers(''.join(x)), books_content))
    # Stack the content of all books on top of each other for form one document
    concat_contents = "\n".join(cleaned_contents)

    return concat_contents

def read_books(books_repo: str) -> List[str]:
    """
    Same as `read_and_concat_books`, but instead of returning a single string of concatenated books, 
    it will return a list of strings where each string is the content of a book.
    """
    # Read all books saved in .txt files
    books = list(iglob(os.path.join(books_repo, "*.txt")))
    books_content = list(map(read_file, books))
    # Remove the GP headers
    cleaned_contents = list(map(lambda x: strip_headers(''.join(x)), books_content))

    return cleaned_contents

def tokenize(text: str):

    words = split_text_into_words(text)

    tokens = []

    for word in words:

        number_tokens, has_number = check_number_token(word)
        if has_number:
            tokens.extend(number_tokens)
            continue

        word_tokens = check_text_token(word)
        if word_tokens:
            tokens.extend(word_tokens)
        
    return tokens

def remove_special_characters(tokens: List) -> List[str]:
    """
    This function will remove special characters.
    """
    pattern = r"^[a-zA-Z]+$|^[+-]?\d+(\.\d*)?$|^[a-zA-Z]+\d+(\.\d*)?$"
    cleaned_tokens = [token for token in tokens if re.match(pattern, token)]
    return cleaned_tokens

def remove_stopwords(tokens: List, stopwords: List = []) -> List[str]:
    """
    This function will remove stopwords.
    """
    stopwords_data = read_file("./data/stopwords.txt")
    stopwords = set(word.strip().lower() for word in stopwords_data)
    cleaned_tokens = [token for token in tokens if token.lower() not in stopwords]
    return cleaned_tokens

def preprocess(text: str) -> List[str]:
    """
    This function will preprocess the text by running a pipline consisting of:
    - Tokenization of text into units (i.e. tokens).
    - Removing special characters and numbers.
    - Removing stopwords.
    """
    functions = [tokenize, remove_stopwords, remove_special_characters]
    tokens = text
    for function in functions:
        tokens = function(tokens)

    return tokens

def check_number_token(word: str) -> List[str]:
    """Processes a word that matches the number pattern."""

    number_pattern = r"^([^\d]*?)([+-]?\d+(\.\d*)?)([^\d]*)$"
    has_number = False

    match = re.search(number_pattern, word)

    tokens = []

    if match:

        # extracts anything that is before and after a number, and further processes that part
        before_number = match.group(1)
        number = match.group(2)
        after_number = match.group(4)

        if before_number:
            text_tokens = check_text_token(before_number)
            tokens.extend(text_tokens)

        if number:
            tokens.append(number)
            has_number = True

        if after_number:
            text_tokens = check_text_token(after_number)
            tokens.extend(text_tokens)

    return tokens, has_number

def check_text_token(word: str) -> List[str]:
    """Processes a word that matches the word pattern."""

    word_pattern = r"^([^a-zA-Z0-9]*)([a-zA-Z'\-_]+)?([^a-zA-Z0-9]*)$"
    contraction_pattern = r"^(.*?)(n't|'\w+)?$"

    match = re.search(word_pattern, word)

    tokens = []

    if match:

        before_text, text, after_text = match.groups()

        tokens.extend(list(before_text))
        
        if text:

            match = re.search(contraction_pattern, text)

            before_apostrophe = match.group(1)
            after_apostrophe = match.group(2)

            tokens.append(before_apostrophe)

            if after_apostrophe:
                tokens.append(after_apostrophe)

        tokens.extend(list(after_text))

    return tokens

def split_text_into_words(text: str) -> List[str]:
    """Splits the text into words by spaces."""
    return re.split(r"\s+", text)