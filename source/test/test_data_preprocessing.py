__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import unittest
from unittest import TestCase

from source.util.data_preprocessing import clean_text, get_dataset, get_features_and_labels


class TestDataPreprocessing(TestCase):

    def test_clean_text_convert_to_lower_case(self):
        """
        Tests text cleaning - conversion to lower case
        """
        self.assertEqual(clean_text("Baishali Dutta"), "baishali dutta")

    def test_clean_text_fix_misspelled_words_PROFLIE_to_PROFILE(self):
        """
        Tests text cleaning - fix misspelled words [PROFLIE -> PROFILE]
        """
        self.assertEqual(clean_text("Conditional Approval Proflie"), "conditional approval profile")

    def test_clean_text_remove_punctuations(self):
        """
        Tests text cleaning - remove punctuations
        """
        self.assertEqual(
            clean_text("Baishali used punctuations;;;;;``´´"), "baishali used punctuations")

    def test_clean_text_remove_stopwords(self):
        """
        Tests text cleaning - remove stopwords
        """
        self.assertEqual(
            clean_text("This dose has been approved very carefully"), "dose approved carefully")

    def test_dataset(self):
        """
        Tests dataset retrieval
        """
        dataset = get_dataset("testdata.xlsx")

        self.assertIsNotNone(dataset)
        self.assertIsNotNone(dataset["Sentiment"])  # new column existence

    def test_dataset_for_positive_sentence(self):
        """
        Tests dataset when 'Positive=1', the associated value in 'Sentiment'
        will be 'Positive'
        """
        dataset = get_dataset("testdata.xlsx")
        self.assertIsNotNone(dataset["Sentiment"])  # new column existence

        self.assertEqual(dataset['Positive'][1], 1)
        self.assertEqual(dataset['Sentiment'][1], "Positive")

    def test_dataset_for_negative_sentence(self):
        """
        Tests dataset when 'Negative=1', the associated value in 'Sentiment'
        will be 'Negative'
        """
        dataset = get_dataset("testdata.xlsx")
        self.assertIsNotNone(dataset["Sentiment"])  # new column existence

        self.assertEqual(dataset['Negative'][8], 1)
        self.assertEqual(dataset['Sentiment'][8], "Negative")

    def test_dataset_for_neutral_sentence(self):
        """
        Tests dataset when 'Neutral=1', the associated value in 'Sentiment'
        will be 'Neutral'
        """
        dataset = get_dataset("testdata.xlsx")
        self.assertIsNotNone(dataset["Sentiment"])  # new column existence

        self.assertEqual(dataset['Neutral'][7], 1)
        self.assertEqual(dataset['Sentiment'][7], "Neutral")

    def test_get_features_and_labels(self):
        """
        Tests features and labels from the dataset
        """
        X, y, vectorizer = get_features_and_labels("testdata.xlsx", 0.1, min_df=1)

        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(vectorizer)

    def test_get_train_test_split_validation(self):
        """
        Tests train test split validation
        """
        X_train, X_test, y_train, y_test = \
            get_features_and_labels("testdata.xlsx", 0.1, min_df=1)

        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)


# -------------------------------------------------------------------------
#                          Main Unit Test Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
