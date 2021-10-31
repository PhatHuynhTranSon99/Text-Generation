import pickle


def save_as_pickle(object, file_name):
    with open(file_name, "wb") as handle:
        pickle.dump(object, handle, pickle.HIGHEST_PROTOCOL)


def load_from_pickle(file_name):
    with open(file_name, "rb") as handle:
        result = pickle.load(handle)
    return result