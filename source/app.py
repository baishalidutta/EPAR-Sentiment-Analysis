__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------

import click

from classifier_factory import get_classifier

# -------------------------------------------------------------------------
#                         Default Configurations
# -------------------------------------------------------------------------
DEFAULT_DATASET_LOC = "../data/sentences_with_sentiment.xlsx"
DEFAULT_TEST_SPLIT = 0.2


# -------------------------------------------------------------------------
#                      Command Line Parseable Function
# -------------------------------------------------------------------------
@click.command()
@click.option('--data', default=DEFAULT_DATASET_LOC, help="Training Data (XLSX) Location")
@click.option('--split', default=DEFAULT_TEST_SPLIT, help="Test Split (in fraction, e.g '0.2')")
@click.option('--classifier', default=1, help="""
                                                \b
                                                    1. Naive Bayes
                                                    2. Decision Tree
                                                    3. Logistic Regression
                                                    4. Random Forest
                                                    5. Linear Support Vector
                                                    6. Kernel Support Vector Machine
                                                    7. XGBoost
                                                    8. Bidirectional LSTM (RNN)""")
def __execute__(data, split, classifier):
    """
    EPAR Sentiment Analysis
    """
    clf = get_classifier(classifier)
    if clf is None:
        raise Exception("Sorry, no classifier found")

    print("======================================================")
    print("                Input Configuration                   ")
    print("======================================================")
    print("1. Data       : " + data)
    print("2. Split      : " + str(split))
    print("3. Classifier : " + clf.name())

    clf.evaluate(data, split)


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    __execute__()
