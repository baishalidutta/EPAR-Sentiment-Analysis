__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.append("..")  # adds higher directory to Python modules path

from source.util.data_preprocessing import get_dataset


class BaseClassifier:
    """
    Contains common functionalities for all classical machine learning classifiers
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def __train__(self, data, split):
        """
        The training operation to perform
        :param data: the data to perform the training on
        :param split: the train-test split on the data
        :return: the text classifier, the independent and dependent test variables
        """
        dataset = get_dataset(data)

        X = dataset['Sentence']
        y = dataset['Sentiment']

        le = LabelEncoder()
        y = le.fit_transform(y)

        vectorizer = TfidfVectorizer(max_features=2500, max_df=0.8, min_df=4)

        X = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

        self.classifier.fit(X_train, y_train)

        return self.classifier, vectorizer, X, y, X_test, y_test

    def evaluate(self, data, split):
        """
        The evaluation operation to perform
        :param data: the data to perform the evaluation on
        :param split: the train-test split on the data
        """
        text_clf, _, X, y, X_test, y_test = self.__train__(data, split)
        predictions = text_clf.predict(X_test)

        print("======================================================")
        print("                  Confusion Matrix                    ")
        print("======================================================")
        print(confusion_matrix(y_test, predictions))

        print("======================================================")
        print("                Classification Report                 ")
        print("======================================================")
        print(classification_report(y_test, predictions))

        print("======================================================")
        print("                    Accuracy Score                    ")
        print("======================================================")
        print(f'{" ":<17} {accuracy_score(y_test, predictions)}')

        # k-fold cross validation
        accuracies = cross_val_score(estimator=text_clf, X=X, y=y, cv=10)

        print("======================================================")
        print("                 Mean Accuracy Score                  ")
        print("======================================================")
        print(f'{" ":<21} {accuracies.mean() * 100:.2f}%')

    def predict(self, data, split, input_text):
        """
        The prediction on single input sentence
        :param data: the data to train on before prediction
        :param split: the test split for training before prediction
        :param input_text: the single sentence to predict the classification on
        :return: the target prediction class
        """
        text_clf, vectorizer, _, _, _, _ = self.__train__(data, split)
        return text_clf.predict(vectorizer.transform([input_text]).toarray())


class DecisionTreeModelClassifier(BaseClassifier):
    """
    The Decision Tree Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Decision Tree"

    def __init__(self):
        super().__init__(DecisionTreeClassifier())


class KernelSvmClassifier(BaseClassifier):
    """
    The Kernel SVM Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Kernel Support Vector Machine"

    def __init__(self):
        super().__init__(SVC(kernel="rbf"))


class LinearSvClassifier(BaseClassifier):
    """
    The Linear Support Vector Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Linear Support Vector"

    def __init__(self):
        super().__init__(LinearSVC())


class LogisticRegressionClassifier(BaseClassifier):
    """
    The Logistic Regression Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Logistic Regression"

    def __init__(self):
        super().__init__(LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000))


class NaiveBayesClassifier(BaseClassifier):
    """
    The Naive Bayes Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Naive Bayes"

    def __init__(self):
        super().__init__(MultinomialNB())


class XgBoostClassifier(BaseClassifier):
    """
    The XGBoost Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "XGBoost"

    def __init__(self):
        super().__init__(XGBClassifier())


class RandomForestModelClassifier(BaseClassifier):
    """
    The Random Forest Classifier Algorithm for training and evaluation
    """

    def name(self):
        return "Random Forest"

    def __init__(self):
        super().__init__(RandomForestClassifier(n_estimators=50))
