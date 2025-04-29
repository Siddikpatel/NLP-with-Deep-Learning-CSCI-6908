from data.dataset import Dataset
from data.textloader import TextLoader
from data.textpreprocessor import TextPreprocessor
from data.vocabulary import Vocabulary
from model.model import CBOWModel
from training.optimizer import SGD
from training.trainer import Trainer

import os


file_path = os.path.join(os.getcwd(), "raw_data", "data.txt")
text = TextLoader.load_text(file_path)

# Preprocess the text
tokens = TextPreprocessor.preprocess(text)

# Build the vocabulary
vocabulary = Vocabulary()
vocabulary.build_vocab(tokens)

# Create the dataset
context_size = 2
dataset = Dataset(tokens, vocabulary, context_size)
print("Vocabulary size:", vocabulary.get_vocab_size())  # Should be the number of unique tokens

# Initialize the CBOW model
vocab_size = vocabulary.get_vocab_size()
embedding_dim = 50
cbow_model = CBOWModel(vocab_size, embedding_dim, context_size)

# Collect all learnable parameters
parameters = [cbow_model.embedding_layer.embeddings, cbow_model.linear_layer.weights]

# Initialize the SGD optimizer
learning_rate = 0.5
optimizer = SGD(parameters, learning_rate)

# Initialize the trainer
trainer = Trainer(cbow_model, optimizer, dataset)

# Train the model
num_epochs = 50
batch_size = 32
trainer.train(num_epochs, batch_size)

# Plot the training loss
trainer.plot_loss()