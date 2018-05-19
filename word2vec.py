import numpy as np
from gensim.models import KeyedVectors

VECTOR_CACHE = 'GoogleNews-vectors-negative300-SLIM.bin'


class Word2VecVectoriser:
    """Uses a small, English only version of the Google News word2vec model
        Developed to be used with the keras embedding layer
    """

    def __init__(self, max_content_length):
        self.max_content_length = max_content_length
        self.vectoriser = KeyedVectors.load_word2vec_format(VECTOR_CACHE, binary=True)
        self.embedding_matrix = self.get_embedding_matrix()

    def get_embedding_matrix(self):
        """Creates a weight matrix for use in a keras embedding layer,
            Maps the index of every word in the vocabulary to its vector representation"""
        embedding_matrix = np.zeros((len(self.vectoriser.vocab), 300))
        for i in range(len(self.vectoriser.vocab)):
            embedding_vector = self.vectoriser[self.vectoriser.index2word[i]]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def vectorise(self, data):
        """Converts each input text to a vector of each word's index in the embedding matrix,
            padded to a fixed length from the right"""
        contents = (x['TEXT'] for x in data)
        vectors = []
        for doc in contents:
            # Each vector is of a fixed length
            vector = np.zeros(self.max_content_length)
            for i, word in enumerate(doc.split(' ')):
                if i == self.max_content_length:
                    break
                # Ignores words not in the vocabulary
                if word in self.vectoriser.vocab:
                    vector[i] = self.vectoriser.vocab[word].index
            vectors.append(vector)
        return np.array(vectors)
