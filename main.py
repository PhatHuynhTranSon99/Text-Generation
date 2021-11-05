from nltk.tokenize import word_tokenize
from scripts.preprocess import generate_ngrams, generate_token_list, generate_word2index_mapping
from scripts.train import NGramsModel
from scripts.utils import softmax
import torch
import random

DATA_PATH = "data/data.txt"
TOKEN_PATH = "checkpoints/token.pickle"
WORD2INDEX_PATH = "checkpoints/word2index.pickle"
NGRAMS_PATH = "checkpoints/ngrams.pickle"
EPOCHS = 20


if __name__ == "__main__":
    # Get tokens, word2index and ngrams
    tokens = generate_token_list(
        data_path=DATA_PATH,
        save_dir=TOKEN_PATH,
        tokenizer=word_tokenize
    )

    word2index = generate_word2index_mapping(
        tokens=tokens,
        save_dir=WORD2INDEX_PATH
    )

    ngrams = generate_ngrams(
        tokens=tokens,
        save_dir=NGRAMS_PATH
    )

    # Creating models and loss
    # Create model
    model = NGramsModel(
        vocab_size=len(word2index),
        embedding_size=50,
        hidden_size=64
    )

    # Create loss and optimizer
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, momentum=0.1)

    # Model to CUDA [If your hardware has this technology]
    # model.cuda()

    # Start training
    EPOCHS = 3
    for epoch in range(EPOCHS):
        # Shuffle n grams
        random.shuffle(ngrams)

        # Save average loss
        avg_loss = 0

        # Stochastic gradient descent
        for i in range(len(ngrams)):
            # Display progress
            if i % 1000 == 0:
                print(f"Iteration: {i}")

            # Unpack to get inputs and labels
            inputs, labels = ngrams[i]
            
            # Convert to indices
            inputs = torch.tensor([word2index[token] for token in inputs], dtype=torch.long)
            labels = torch.tensor([word2index[token] for token in labels], dtype=torch.long)

            # Inputs and labels to cuda
            #inputs = inputs.to("cuda:0")
            #labels = labels.to("cuda:0")

            # Reset optimizer
            optimizer.zero_grad()

            # Run through model and get the loss
            result = model(inputs)
            result = result.view(result.shape[1], result.shape[2]) # Remove the extra dimension by GRU layer
            current_loss = loss(result, labels)

            # Add to average loss
            avg_loss += current_loss.item()

            # Perform backprop
            current_loss.backward()
            optimizer.step()

        # Calculate and display current loss
        avg_loss = avg_loss / len(ngrams)
        print(f"Epoch: {epoch + 1}, Average loss: {avg_loss}")

    # Save model and evaluate
    torch.save(model.state_dict(), "model.pth")

    # Create index to word mapping
    index2word = { v: k for k, v in word2index.items() }

    # Prediction phase
    prompt = ["i", "should", "be", "there"]
    completed = prompt + []

    # Run prompt though model
    with torch.no_grad():
        for i in range(80):
            # Convert to indices
            inputs = torch.tensor([word2index[token] for token in prompt], dtype=torch.long)
            inputs = inputs.to("cuda:0")

            # Predict next words
            result = model(inputs)
            result = result.view(result.shape[1], result.shape[2]) # Remove the extra dimension by GRU layer

            # Test print
            probs = result[-1].cpu().detach().numpy()
            prediction = np.random.choice(
                a=len(word2index),
                p=softmax(probs)
            )

            completed.append(index2word[prediction])

            # Concat to prompt
            prompt = prompt[1:]
            prompt.append(index2word[prediction])

    # Print the complete sentence
    print(" ".join(completed))