__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import sys

import click

sys.path.append("..")  # adds higher directory to Python modules path

from source.util.classifier_factory import get_classifier
from source.util.grid_search import grid_search

# -------------------------------------------------------------------------
#                         Default Configurations
# -------------------------------------------------------------------------
DEFAULT_DATASET_LOC = "../data/sentences_with_sentiment.xlsx"
DEFAULT_TEST_SPLIT = 0.2
"""
The number 36 for cross-validation denotes that k-folds will be used 
with 36 splits.

The number 36 has been chosen since sklearn generates warning for any 
value greater than 36. According to stratified K-folds, the warning gets 
generated if the number of elements in the least populated class is 
less than the number of splits. In our dataset, the least populated 
class (Negative labelled) contains 36 items (sentences) and that's why 
36 has been chosen as the number of splits in K-folds.
"""
DEFAULT_CV_SPLIT = 36


# -------------------------------------------------------------------------
#                   Command Line Interface (CLI) Function
# -------------------------------------------------------------------------
@click.command()
@click.option('--data', default=DEFAULT_DATASET_LOC, help="Training Data (XLSX) Location")
@click.option('--testsplit', default=DEFAULT_TEST_SPLIT, help="Test Split (in fraction, e.g '0.2')")
@click.option('--cvsplit', default=DEFAULT_CV_SPLIT, help="Cross-Validation Split (in integer, e.g '36')")
@click.option('--grid', default=False, help="Perform Grid Search", is_flag=True)
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
def __execute__(data, testsplit, cvsplit, grid, classifier):
    """
    EPAR Sentiment Analysis
    """
    if grid:
        __display_input_configuration__(data, None, cvsplit, None)
        grid_search(data, cvsplit)
    else:
        clf = get_classifier(classifier)
        if clf is None:
            raise Exception("Sorry, no classifier found")

        __display_input_configuration__(data, testsplit, cvsplit, clf.name[0])
        clf.evaluate(data, testsplit, cvsplit)


def __display_input_configuration__(data, testsplit, cvsplit, classifier_name):
    """
    Displays the CLI input configuration

    :param data: the data location
    :param testsplit: the test split
    :param cvsplit: the cross-validation split
    :param classifier_name: the name of the classifier
    """
    i = 1
    print("======================================================")
    print("                Input Configuration                   ")
    print("======================================================")
    print(f"{i}. Data                   : {data}")  # data is always displayed

    if testsplit is not None:
        i += 1
        print(f"{i}. Test Split             : {testsplit}")

    if cvsplit is not None:
        i += 1
        print(f"{i}. Cross-Validation Split : {cvsplit}")

    if classifier_name is not None:
        i += 1
        print(f"{i}. Classifier             : {classifier_name}")


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    __execute__()
