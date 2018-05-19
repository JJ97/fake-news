from sklearn.naive_bayes import MultinomialNB


class NaiveBayesClassifier:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.classifier = MultinomialNB()

    def __str__(self):
        return 'naive bayes'

    def train(self, train_set, validation_set=None):
        vectors = self.vectorizer.vectorise(train_set)
        labels = [int(x['LABEL']) for x in train_set]
        self.classifier.fit(vectors, labels)

    def predict(self, test_set):
        test_vectors = self.vectorizer.vectorise(test_set)
        predictions = self.classifier.predict(test_vectors)
        return predictions
