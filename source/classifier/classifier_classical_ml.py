__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from source.classifier.classifier import Classifier
from source.util.data_preprocessing import get_features_and_labels, get_train_test_split_validation


class ClassicalMLClassifierBase(Classifier):
    """
    Contains common implementations for all classical machine learning classifiers
    """

    def __init__(self, name, classifier, hyperparams):
        """
        The constructor for classical machine learning classifier

        :param name: the name of the classifier
        :param classifier: the sklearn classifier instance
        :param hyperparams: the hyperparameters to be tuned in grid search
        """
        self.name = name,
        self.classifier = classifier
        self.hyperparams = hyperparams

    def evaluate(self, data, testsplit, cvsplit):
        """
        The common implementation of the evaluation process. Since the dataset
        is small, we don't need to perform training and the evaluation operations
        separately. We can simply perform the evaluation operation which would first
        perform the training and thereafter evaluation on the validation set.

        :param data: the data to perform the evaluation on
        :param testsplit: the test split on the data
        :param cvsplit: the cross-validation split on the data
        """
        print("======================================================")
        print("    >>  Performing Train/Test Split Validation  <<    ")
        print("======================================================")

        X, y, vectorizer = get_features_and_labels(data)
        X_train, X_test, y_train, y_test = get_train_test_split_validation(data, testsplit)

        # store vectorizer in a class variable to use it during prediction of a single sentence
        self.vectorizer = vectorizer
        self.classifier.fit(X_train, y_train)

        predictions = self.classifier.predict(X_test)

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
        print(f'{" ":<21} {accuracy_score(y_test, predictions) * 100:.2f}%')

        print("======================================================")
        print("          >>  Performing Cross-Validations  <<        ")
        print("======================================================")

        print("======================================================")
        print("           Performing K-Folds Cross-Validation        ")
        print("======================================================")

        """
        k-folds cross-validation - this will internally fit the model and
        the previous co-efficients, weights, bias will be reset in the 
        classifier
        """
        accuracies = cross_val_score(estimator=self.classifier, X=X, y=y, cv=cvsplit)

        print("======================================================")
        print("                 Mean Accuracy Score                  ")
        print("======================================================")
        print(f'{" ":<21} {accuracies.mean() * 100:.2f}%')

        print("======================================================")
        print("       Performing Leave One Out Cross-Validation      ")
        print("======================================================")

        """
        Leave one out cross-validation - this will internally fit the model
        once again and the previous co-efficients, weights, bias will be reset 
        in the classifier
        """
        accuracies = cross_val_score(estimator=self.classifier, X=X, y=y, cv=LeaveOneOut())

        print("======================================================")
        print("                 Mean Accuracy Score                  ")
        print("======================================================")
        print(f'{" ":<21} {accuracies.mean() * 100:.2f}%')

    def predict(self, data, testsplit, cvsplit, input_sentence):
        """
        The prediction on single input sentence

        :param data: the data to train on before prediction
        :param testsplit: the test split for training before prediction
        :param cvsplit: the cross-validation split for training before prediction
        :param input_sentence: the single sentence to predict the classification on
        :return: the target prediction class
        """
        """
        Fit the model before performing prediction. The model will be fitted using leave
        one out cross validation since it is applied at the very end and the final fitted
        model will then be used for prediction.
        """
        self.evaluate(data, testsplit, cvsplit)
        return self.classifier.predict(self.vectorizer.transform([input_sentence]).toarray())


# -------------------------------------------------------------------------
#             Concrete Classical ML Classifier Implementations
# -------------------------------------------------------------------------

class DecisionTreeModelClassifier(ClassicalMLClassifierBase):
    """
    The Decision Tree Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random']
        }
        super().__init__("Decision Tree", DecisionTreeClassifier(), hyperparams)


class KernelSvmClassifier(ClassicalMLClassifierBase):
    """
    The Kernel SVM Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = [
            {
                'C': [1, 10, 100, 1000],
                'kernel': ['linear']
            },
            {
                'C': [1, 10, 100, 1000],
                'kernel': ['rbf'],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        ]
        super().__init__("Kernel Support Vector Machine", SVC(kernel="rbf"), hyperparams)


class LinearSvClassifier(ClassicalMLClassifierBase):
    """
    The Linear Support Vector Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {'C': [1, 10, 100, 1000]}
        super().__init__("Linear Support Vector", LinearSVC(max_iter=10000), hyperparams)


class LogisticRegressionClassifier(ClassicalMLClassifierBase):
    """
    The Logistic Regression Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {
            'C': [0.1, 1, 10, 100, 1000],
            'solver': ['newton-cg', 'sag', 'lbfgs']
        }
        super().__init__("Logistic Regression", LogisticRegression(max_iter=10000), hyperparams)


class NaiveBayesClassifier(ClassicalMLClassifierBase):
    """
    The Naive Bayes Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {'alpha': [0.001, 0.1, 1, 10, 100]}
        super().__init__("Naive Bayes", MultinomialNB(), hyperparams)


class RandomForestModelClassifier(ClassicalMLClassifierBase):
    """
    The Random Forest Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 50, 100],
            'n_estimators': [10, 50, 100, 150]
        }
        super().__init__("Random Forest", RandomForestClassifier(n_estimators=50), hyperparams)


class XgBoostClassifier(ClassicalMLClassifierBase):
    """
    The XGBoost Classifier Algorithm for training and evaluation
    """

    def __init__(self):
        hyperparams = {
            'n_estimators': [10, 50, 100, 200]
        }
        super().__init__("XGBoost", XGBClassifier(), hyperparams)
