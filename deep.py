import numpy as np
from keras import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, SimpleRNN

LSTM_WEIGHTS = 'weights_lstm.hdf5'
RNN_WEIGHTS = 'weights_rnn.hdf5'


class DeepClassifier:
    """Keras neural net using an embedding layer and  a single LSTM or RNN layer"""
    def __init__(self, vectorizer, use_lstm=True, layer_size=8, batch_size=32, learning_rate=0.005,
                 dropout=0.02, use_cache=True):
        self.vectorizer = vectorizer
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.use_lstm = use_lstm
        self.use_cache = use_cache
        self.classifier = self.initialise_classifier()

    def __str__(self):
        description = '{} layer_size:{} batch_size:{} learning_rate:{} dropout:{}'
        layer_type = 'lstm' if self.use_lstm else 'rnn'
        return description.format(layer_type, self.layer_size, self.batch_size, self.learning_rate, self.dropout)

    def initialise_classifier(self):
        """"Sets up the keras model with user-specified parameters and embedding layer,
            Loads weights from a cache if specified by the user
        """
        # Input is an embedding layer using the weights from the pre-trained word2vec vectoriser
        embedding_matrix = self.vectorizer.embedding_matrix
        vocab_size, embedding_vector_length = embedding_matrix.shape

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
                            trainable=False, mask_zero=True))

        # Add either an LSTM or RNN layer with the same parameters
        if self.use_lstm:
            model.add(LSTM(self.layer_size, dropout=self.dropout, recurrent_dropout=0.2))
        else:
            model.add(SimpleRNN(self.layer_size, dropout=self.dropout, recurrent_dropout=0.2))

        # Output layer
        model.add(Dense(1, activation='sigmoid'))

        # Load pre-trained weights if specified by user
        if self.use_cache:
            weight_file = LSTM_WEIGHTS if self.use_lstm else RNN_WEIGHTS
            model.load_weights(weight_file)

        optimizer = optimizers.adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self, train_set, validation_set=np.array([])):
        """Trains the classifier for max 100 epochs on a given training and validation set"""
        x_train = self.vectorizer.vectorise(train_set)
        y_train = np.array([int(x['LABEL']) for x in train_set])

        x_validate = self.vectorizer.vectorise(validation_set)
        y_validate = np.array([int(x['LABEL']) for x in validation_set])

        # Stops training after 5 epochs with no improvement in validation loss
        monitor = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

        self.classifier.fit(x_train, y_train, batch_size=min(self.batch_size, len(train_set)),
                            epochs=100, verbose=2, validation_data=(x_validate, y_validate),
                            callbacks=[monitor])

    def predict(self, test_set):
        """"Classifies a given set of texts"""
        x_test = self.vectorizer.vectorise(test_set)
        predictions = self.classifier.predict(x_test)
        predictions = [int(x > 0.5) for x in predictions]
        return predictions
