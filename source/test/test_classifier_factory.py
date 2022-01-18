__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import unittest
from unittest import TestCase

from source.classifier.classifier_classical_ml import KernelSvmClassifier, DecisionTreeModelClassifier, \
    LinearSvClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, RandomForestModelClassifier, \
    XgBoostClassifier
from source.classifier.classifier_lstm import LstmClassifier
from source.util.classifier_factory import get_classifier


class TestClassifierFactory(TestCase):

    def test_get_classifier_for_valid_input(self):
        """
        Tests all valid inputs
        """
        self.assertTrue(isinstance(get_classifier(1), NaiveBayesClassifier))
        self.assertTrue(isinstance(get_classifier(2), DecisionTreeModelClassifier))
        self.assertTrue(isinstance(get_classifier(3), LogisticRegressionClassifier))
        self.assertTrue(isinstance(get_classifier(4), RandomForestModelClassifier))
        self.assertTrue(isinstance(get_classifier(5), LinearSvClassifier))
        self.assertTrue(isinstance(get_classifier(6), KernelSvmClassifier))
        self.assertTrue(isinstance(get_classifier(7), XgBoostClassifier))
        self.assertTrue(isinstance(get_classifier(8), LstmClassifier))

    def test_get_classifier_for_invalid_input(self):
        """
        Tests invalid input
        """
        self.assertIsNone(get_classifier(0))


# -------------------------------------------------------------------------
#                          Main Unit Test Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
