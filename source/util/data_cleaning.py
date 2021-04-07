__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import re

import nltk
import spacy
from nltk.corpus import stopwords

# -------------------------------------------------------------------------
#                           One-time Instances
# -------------------------------------------------------------------------

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nltk.download('stopwords')
stop_words = stopwords.words('english')


# -------------------------------------------------------------------------
#                           Data Cleaning
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
    Fixes the misspelled words on the specified text (uses predefined misspelled dictionary)
    :param text: The text to be fixed
    :return: the fixed text
    """
    mispelled_dict = {'proflie': 'profile'}
    for word in mispelled_dict.keys():
        text = text.replace(word, mispelled_dict[word])
    return text


def __remove_punctuations__(text):
    """
    Removes all punctuations from the specified text
    :param text: the text whose punctuations to be removed
    :return: the text after removing the punctuations
    """
    return re.sub(r'[^\w\s]', '', text)


def __remove_stopwords__(text):
    """
    Removes all stop words from the specified text
    :param text: the text whose stop words need to be removed
    :return: the text after removing the stop words
    """
    return " ".join(x for x in text.split() if x not in stop_words)


def __lemmatise__(text):
    """
    Lemmatises the specified text
    N.B: Lemmatisation has not been applied since its
    usage in sentiment analysis is debatable as it disrupts
    part of the speech tagging
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
    :param text: the text which needs to be cleaned
    :return: the cleaned text
    """
    text = __convert_to_lower_case__(text)
    text = __fix_misspelled_words__(text)
    text = __remove_punctuations__(text)
    text = __remove_stopwords__(text)

    return text
