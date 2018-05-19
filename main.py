print("IMPORTING MODULES")

import csv
import os
import pickle
import random
import re
import string

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support

from bayes import NaiveBayesClassifier
from deep import DeepClassifier
from tfidf import TfIdfVectoriser
from word2vec import Word2VecVectoriser

USE_CACHE = True
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
SANITISED_CORPUS_CACHE = "{}/{}".format(APP_ROOT, "sanitised_corpus2.p")
CORPUS_FILE_NAME = 'news_ds.csv'

MAX_SEQUENCE_LENGTH = 1000
STOPWORDS = set(stopwords.words("english"))

# Ensures that training, validation, testing set are randomly sampled but the same between each run
random.seed(0x5EED)


def sanitise(dirty):
    """Takes a single text from the corpus and converts it to a standard form for better vectorisation results"""
    # Removes html tags
    clean = BeautifulSoup(dirty, "html.parser").get_text()
    # Converts to ascii, removing characters that cannot be converted
    clean = clean.encode('ascii', 'ignore').decode("ascii")
    # Converts to lowercase and removes non-alphanumeric characters besides space and hyphen
    clean = ''.join(c for c in clean.lower() if c in string.ascii_letters
                    or c in string.digits
                    or c.isspace() or c == '-')
    # Removes newlines and multiple consecutive hyphens
    clean = clean.replace('\n', ' ')
    clean = re.sub('([-]{2,})', '', clean)
    # Shrinks consecutive spaces to a single occurrence
    clean = re.sub('([ ]{2,})', ' ', clean)
    # Strips whitespace from both sides
    clean = clean.strip()
    # Filters out stopwords using nltk list
    clean = ' '.join(word for word in clean.split(" ") if word not in STOPWORDS)
    return clean


def read_data():
    """Loads the corpus from cache if USE_CACHE flag == True
        otherwise, loads corpus, sanitises text and splits into training, validation and testing sets"""
    if USE_CACHE:
        return pickle.load(open(SANITISED_CORPUS_CACHE, "rb"))

    print('LOADING CORPUS\n')
    with open(CORPUS_FILE_NAME) as dataset:
        reader = csv.DictReader(dataset)
        # Keeps track of seen text for filtering out duplicates.
        already_seen_text = set()
        dataset = []
        for row in reader:
            dirty_text = row['TEXT']
            # Converts text to a standard form for better vectorisation results.
            clean_text = sanitise(dirty_text)
            # Filters out meaninglessly small and already seen texts.
            if len(clean_text) > 10 and clean_text not in already_seen_text:
                row['TEXT'] = clean_text
                dataset.append(row)
                already_seen_text.add(clean_text)

    # Splits corpus by 80/20 train/test and training set by 80/20 train/validate.
    corpus_size = len(dataset)
    train_size = int(corpus_size * 0.8 * 0.8)
    validation_size = int(corpus_size * 0.8 * 0.2)

    random.shuffle(dataset)
    train_set = dataset[:train_size]
    validation_set = dataset[train_size: train_size + validation_size]
    test_set = dataset[train_size + validation_size:]


    pickle.dump((train_set, validation_set, test_set), open(SANITISED_CORPUS_CACHE, "wb"))
    return train_set, validation_set, test_set


def evaluate(classifier, test_set):
    """Evaluates a classifier with respect to its predictions on a given test set"""
    predictions = classifier.predict(test_set)
    ground_truth = [int(x['LABEL']) for x in test_set]
    p, r, f1, s = precision_recall_fscore_support(ground_truth, predictions)
    print('      PRECISION: {}'.format(p))
    print('         RECALL: {}'.format(r))
    print('             F1: {}'.format(f1))
    print('        SUPPORT: {}\n'.format(s))


def main():
    train_set, validation_set, test_set = read_data()

    print("  TRAINING SET SIZE: {}".format(len(train_set)))
    print("VALIDATION SET SIZE: {}".format(len(validation_set)))
    print("   TESTING SET SIZE: {}\n".format(len(test_set)))

    print("FITTING tf-idf vectoriser")
    tfidf = TfIdfVectoriser(train_set, use_idf=False, df_range=(0.1, 1.0), ngram_range=(1, 2))

    print("LOADING Word2Vec vectoriser with max sequence length {}".format(MAX_SEQUENCE_LENGTH))
    w2v_slim = Word2VecVectoriser(MAX_SEQUENCE_LENGTH)
    print("INITIALISING CLASSIFIERS")
    dropout, learning_rate, batch_size, layer_size = (0.2, 0.005, 32, 8)
    lstm = DeepClassifier(w2v_slim, True, layer_size, batch_size, learning_rate, dropout, USE_CACHE)
    rnn = DeepClassifier(w2v_slim, False, layer_size, batch_size, learning_rate, dropout, USE_CACHE)
    bayes = NaiveBayesClassifier(tfidf)

    print("TESTING CLASSIFIERS")
    for classifier in (bayes, lstm, rnn):
        # Naive bayes is not cached so must always be trained
        if not USE_CACHE or isinstance(classifier, NaiveBayesClassifier):
            print('   TRAINING {}'.format(classifier), flush=True)
            classifier.train(train_set, validation_set=validation_set)
        print('   TESTING {}'.format(classifier), flush=True)
        evaluate(classifier, test_set)


if __name__ == "__main__":
    main()
