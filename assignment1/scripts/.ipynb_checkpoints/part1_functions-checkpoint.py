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


def tokenize(text: str):
    tokens = re.split(r'\s+', text)


def remove_special_characters(tokens: List) -> List[str]:
    """
    This function will remove special characters.
    """
    ...

def remove_stopwords(tokens: List, stopwords: List = []) -> List[str]:
    """
    This function will remove stopwords.
    """
    ...

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