__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from abc import ABC


class Classifier(ABC):
    """
    Abstract class to represent a classifier. Every classifier should
    conform to this.
    """

    def evaluate(self, data, testsplit, cvsplit):
        """
        The evaluation operation to perform. Since the dataset
        is small, we don't need to perform training and the evaluation operations
        separately. We can simply perform the evaluation operation which would first
        perform the training and thereafter evaluation on the validation set.

        :param data: the data to perform the evaluation on
        :param testsplit: the test split on the data
        :param cvsplit: the cross-validation split on the data
        """
        pass

    def predict(self, data, testsplit, cvsplit, input_text):
        """
        The prediction on input sentence

        :param data: the data to train on before prediction
        :param testsplit: the test split for training before prediction
        :param cvsplit: the cross-validation split for training before prediction
        :param input_text: the single sentence to predict the classification on
        :return: the target prediction class
        """
        pass
