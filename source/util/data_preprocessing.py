__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import re

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------------
#                           One-time Instances
# -------------------------------------------------------------------------

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nltk.download('stopwords')
stop_words = stopwords.words('english')


# -------------------------------------------------------------------------
#                       Data Cleaning Utility Methods
# -------------------------------------------------------------------------

def __convert_to_lower_case__(text):
    """
    Coverts the specified text to lower case

    :param text: the text to convert
    :return: the lower cased text
    """
    return " ".join(text.lower() for text in text.split())


def __fix_misspelled_words__(text):
    """
    Fixes the misspelled words in the specified text
    (uses predefined misspelled dictionary)

    :param text: The text to be fixed
    :return: the fixed text
    """
    mispelled_dict = {'proflie': 'profile'}
    for word in mispelled_dict.keys():
        text = text.replace(word, mispelled_dict[word])
    return text


def __remove_punctuations__(text):
    """
    Removes all punctuations in the specified text.
    It matches character(s) that is/are not word character(s)
    or spaces and replaces them with empty strings.

    :param text: the text whose punctuations to be removed
    :return: the text after removing the punctuations
    """
    return re.sub(r'[^\w\s]', '', text)


def __remove_stopwords__(text):
    """
    Removes all stop words in the specified text

    :param text: the text whose stop words need to be removed
    :return: the text after removing the stop words
    """
    return " ".join(x for x in text.split() if x not in stop_words)


def __lemmatise__(text):
    """
    Lemmatises the specified text

    N.B: Lemmatisation has not been applied since its
    usage in sentiment analysis is debatable as it disrupts
    part of the speech tagging. It also alters the polarity
    of the words.

    :param text: the text which needs to be lemmatised
    :return: the lemmatised text
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def clean_text(text):
    """
    Cleans the specified text

    The cleaning procedure is as follows:

    1. Convert the context to lower case
    2. Fix misspelled words
    3. Remove all punctuations
    4. Remove all stop words

    :param text: the text to be cleaned
    :return: the cleaned text
    """
    text = __convert_to_lower_case__(text)
    text = __fix_misspelled_words__(text)
    text = __remove_punctuations__(text)
    text = __remove_stopwords__(text)

    return text


# -------------------------------------------------------------------------
#                         Dataset Utility Methods
# -------------------------------------------------------------------------

def get_dataset(dataset_location, sheet_name="Sheet1"):
    """
    Returns the dataset

    :param dataset_location: the dataset location
    :param sheet_name: the name of the sheet
    :return: the dataset with the sentiment column added
    """
    dataset = pd.read_excel(
        dataset_location,
        sheet_name=sheet_name,
        index_col=0)
    return __add_sentiment_column__(dataset)


def __add_sentiment_column__(dataset):
    """
    Create a new column for multi-class classification

    :param dataset: the dataset to use
    :return: the updated dataset
    """
    dataset['Sentiment'] = dataset['Positive'] * 100 + dataset['Negative'] * -100
    dataset['Sentiment'] = dataset['Sentiment'].map({100: 'Positive', -100: 'Negative', 0: 'Neutral'})

    return dataset


def get_features_and_labels(data, min_df=4):
    """
    Returns features and labels from the specified data

    :param data: the dataset to use for feature extraction
    :param min_df: minimum document frequency for TF-IDF vectorizer
    :return: the features and labels from the dataset and the used vectorizer
    """
    dataset = get_dataset(data)

    X = dataset['Sentence']
    y = dataset['Sentiment']

    le = LabelEncoder()
    y = le.fit_transform(y)

    vectorizer = TfidfVectorizer(max_features=2500, max_df=0.8, min_df=min_df)

    X = vectorizer.fit_transform(X)

    return X, y, vectorizer


def get_train_test_split_validation(data, split, min_df=4):
    """
    Returns train test split subsets

    :param data: the dataset to use for feature extraction
    :param split: the test split to use for extraction
    :return: train-test split of inputs
    """
    X, y, _ = get_features_and_labels(data, min_df)
    return train_test_split(X, y, test_size=split)
