import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import time

start = time.time()
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
# train = train.sample(100)


def review_to_words(raw_review):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    # a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(meaningful_words))

num_reviews = train["review"].size
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train["review"][i] ) )
print(time.time() - start)