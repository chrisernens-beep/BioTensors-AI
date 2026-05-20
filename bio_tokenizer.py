import numpy as np


class BioTokenizer:
    """Encode text as fixed spike patterns for sensory input."""

    def __init__(self, input_size=100):
        self.input_size = input_size
        self.vocab = {}
        self.reverse_vocab = {}

    def register_word(self, word, pattern):
        pattern = np.asarray(pattern, dtype=np.float32)
        self.vocab[word] = pattern
        self.reverse_vocab[tuple(pattern.astype(int))] = word

    def encode(self, word):
        """Transform a word into a unique, reproducible spike pattern."""
        if word not in self.vocab:
            rng = np.random.default_rng(abs(hash(word)) % (2**32))
            pattern = np.zeros(self.input_size, dtype=np.float32)
            n_active = min(10, self.input_size)
            indices = rng.choice(self.input_size, size=n_active, replace=False)
            pattern[indices] = 1.0
            self.register_word(word, pattern)
        return self.vocab[word]

    def decode(self, pattern):
        """Best-effort lookup from a spike pattern back to a word."""
        key = tuple(np.asarray(pattern).astype(int))
        return self.reverse_vocab.get(key)
