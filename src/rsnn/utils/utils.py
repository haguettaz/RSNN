import os
import pickle


def save(obj, filename):
    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)