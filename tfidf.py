from sklearn.feature_extraction.text import TfidfVectorizer
from os import path

class TfIdfVectoriser:
    """Essentially a wrapper around the scikit-learn vectoriser,
        allowing for it to be used with the same method calls as the Word2Vec vectoriser
    """
    def __init__(self, fit_data, use_idf, df_range=(0.0, 1.0), ngram_range=(1, 1)):
        self.df_range = df_range
        self.ngram_range = ngram_range

        # Initialise a TfidfVectorizer from sklearn and fit the model
        self.vectoriser = TfidfVectorizer(use_idf=use_idf, ngram_range=ngram_range,
                                          min_df=df_range[0], max_df=df_range[1])

        contents = (x['TEXT'] for x in fit_data)
        self.vectoriser.fit(contents)


    def vectorise(self, data):
        contents = (x['TEXT'] for x in data)
        return self.vectoriser.transform(contents)
