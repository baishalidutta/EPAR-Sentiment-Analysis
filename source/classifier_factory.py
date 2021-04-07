__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from classifier.classifiers import *
from classifier.lstm_classifier import LstmClassifier

classifiers = {
    1: NaiveBayesClassifier(),
    2: DecisionTreeModelClassifier(),
    3: LogisticRegressionClassifier(),
    4: RandomForestModelClassifier(),
    5: LinearSvClassifier(),
    6: KernelSvmClassifier(),
    7: XgBoostClassifier(),
    8: LstmClassifier()
}


def get_classifier(id):
    """
    Returns the classifier by the specified id
    :param id: the id of the classifier
    :return: the model classifier to return
    """
    return classifiers.get(int(id))
