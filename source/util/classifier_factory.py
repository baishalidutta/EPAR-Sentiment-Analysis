__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from source.classifier.classifier_classical_ml import DecisionTreeModelClassifier, KernelSvmClassifier, \
    LinearSvClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, RandomForestModelClassifier, \
    XgBoostClassifier
from source.classifier.classifier_lstm import LstmClassifier

classical_ml_classifiers = {
    1: NaiveBayesClassifier(),
    2: DecisionTreeModelClassifier(),
    3: LogisticRegressionClassifier(),
    4: RandomForestModelClassifier(),
    5: LinearSvClassifier(),
    6: KernelSvmClassifier(),
    7: XgBoostClassifier()
}

deep_learning_classifiers = {
    8: LstmClassifier()
}

# merge all classifiers
all_classifiers = {**classical_ml_classifiers, **deep_learning_classifiers}


def get_classifier(clf_id):
    """
    Returns the classifier by the specified id

    :param clf_id: the id of the classifier
    :return: the model classifier to return
    """
    return all_classifiers.get(int(clf_id))
