import torch

class NGramsModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        # Call constructor of parents
        super(NGramsModel, self).__init__()

        # Cache the sizes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Create the embedding layer
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_size)

        # Create gru layer
        self.gru = torch.nn.GRU(embedding_size, hidden_size, batch_first=True)

        # Create hidden dense layer
        self.dense = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        # Get the embedding from inputs
        embeddings = self.embeddings(inputs)

        # Expand embedding by adding an extra dimension
        embeddings = embeddings.view(1, embeddings.shape[0], embeddings.shape[1])

        # Pass embedding as input into gru and get 
        # Hidden state for each word in the sentence
        hidden_states, _ = self.gru(embeddings)

        # Pass h_t into linear layer and get the result
        result = self.dense(hidden_states)
        return result
