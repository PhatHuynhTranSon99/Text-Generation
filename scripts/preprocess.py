from nltk.tokenize import word_tokenize
from scripts.utils import save_as_pickle


def tokenize_line(tokenizer, line):
    '''
    Process a line into token, apply lowercase if possible
    '''
    words = tokenizer(line)
    return lower_case_words(words)


def lower_case_words(words):
    '''
    Turn every string in a list into its lowercase version
    '''
    return [word.lower() for word in words]


def generate_token_list(data_path, tokenizer, save_dir, save=True):
    '''
    Generate one long list of sentences containings the content of the data
    Save the data list as pickle if required
    '''
    # Hold the tokens
    tokens = []

    with open(data_path, mode="r", encoding="utf-8", errors="ignore") as text_file:
        for line in text_file:
            # Check if line is not empty
            if len(line) > 0:
                # Split the line into tokens
                # Add new line to indicate new line
                line_token = tokenize_line(tokenizer, line) + ["\n"]
                # Append to tokens
                tokens += line_token

    # Save the token list if required
    if save:
        save_as_pickle(tokens, save_dir)

    return tokens


def generate_word2index_mapping(tokens, save_dir, save=True):
    '''
    Create mapping from word to index from token list
    Save the tokens if specified
    '''
    word2index = {}
    current_index = 0

    for token in tokens:
        if token not in word2index:
            word2index[token] = current_index
            current_index += 1

    if save:
        save_as_pickle(word2index, save_dir)

    return word2index