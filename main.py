from nltk import data, tokenize
from nltk.tokenize import word_tokenize
from scripts.preprocess import generate_token_list, generate_word2index_mapping

DATA_PATH = "data/data.txt"
TOKEN_PATH = "checkpoints/token.pickle"
WORD2INDEX_PATH = "checkpoints/word2index.pickle"

tokens = generate_token_list(
    data_path=DATA_PATH,
    save_dir=TOKEN_PATH,
    tokenizer=word_tokenize
)

word2index = generate_word2index_mapping(
    tokens=tokens,
    save_dir=WORD2INDEX_PATH
)

print(word2index)