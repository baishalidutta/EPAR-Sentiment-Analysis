__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021-2022 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import operator

from sklearn.model_selection import GridSearchCV, LeaveOneOut

from source.util.classifier_factory import classical_ml_classifiers
from source.util.data_preprocessing import get_features_and_labels


def grid_search(data, cv):
    """
    Performs grid search on all currently available classical
    machine learning classifiers in 'classifier' directory
    to find out the best performing model. This internally
    executes grid search with K-Folds cross-validation splitting
    strategy as well as Leave One Out cross-validation splitting
    strategy.

    :param data: the data to extract the features and labels from
    :param cv: cross-validation splitting strategy
    """
    print("======================================================")
    print("       Performing Grid Search with K-Folds CV         ")
    print("======================================================")
    grid_search_with_cv(data, cv)

    print("======================================================")
    print("     Performing Grid Search with Leave One Out CV     ")
    print("======================================================")
    grid_search_with_cv(data, LeaveOneOut())


def grid_search_with_cv(data, cv):
    """
    Performs grid search on all currently available classical
    machine learning classifiers in 'classifier' directory
    to find out the best performing model

    :param data: the data to extract the features and labels from
    :param cv: cross-validation splitting strategy
    """

    result = []

    for clf in classical_ml_classifiers.values():
        # associated sklearn classifier
        classifier = clf.classifier

        X, y, _ = get_features_and_labels(data)

        # grid search
        grid = GridSearchCV(estimator=classifier,
                            param_grid=clf.hyperparams,
                            cv=cv,
                            n_jobs=-1)
        grid.fit(X, y)

        print("======================================================")
        print(f"         Best Performing Model for {clf.name[0]}         ")
        print("======================================================")
        print(grid.best_params_)
        print(f"Best Score: {grid.best_score_ * 100:.2f}%")

        # store result
        result.append({'grid': grid,
                       'classifier': grid.best_estimator_,
                       'best score': grid.best_score_,
                       'best params': grid.best_params_,
                       'cv': grid.cv})

    # sort result by best score
    result = sorted(result, key=operator.itemgetter('best score'), reverse=True)

    print("======================================================")
    print("              Best Performing Classifier              ")
    print("======================================================")
    print(result[0]['grid'])
