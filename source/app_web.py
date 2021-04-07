__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import gradio as gr
import numpy as np

from app import DEFAULT_DATASET_LOC, DEFAULT_TEST_SPLIT
from classifier_factory import get_classifier, classifiers


# -------------------------------------------------------------------------
#                           Main Application
# -------------------------------------------------------------------------

def __make_prediction__(input_classifier, input_sentence):
    """
    Predicts the polarity of the specified sentence
    :param input_classifier: the classifier to use
    :param input_sentence: the sentence to be verified
    :return: the polarity
    """
    clf = get_classifier(input_classifier + 1)
    if clf is None:
        raise Exception("Sorry, no classifier found")

    result = clf.predict(DEFAULT_DATASET_LOC, DEFAULT_TEST_SPLIT, input_sentence)
    return __match_class__(result)


def __match_class__(result):
    res = np.asarray(result)
    if len(res.shape) == 2:
        # LSTM result
        if result[0][0] > 0.5:
            return "Positive"
        elif result[0][1] > 0.5:
            return "Negative"
        else:
            return "Neutral"
    else:
        # classical machine learning classifier result
        if result[0] == 0:
            return "Negative"
        elif result[0] == 1:
            return "Neutral"
        else:
            return "Positive"


# retrieve the names of the classifiers from factory
index, values = zip(*classifiers.items())
classifier_list = map(lambda clf: clf.name(), values)

classifier = gr.inputs.Dropdown(list(classifier_list), label="Classifier", type="index")
sentence = gr.inputs.Textbox(lines=17, placeholder="Enter your sentence here")

title = "EPAR Sentiment Analysis"
description = "This application uses several classifiers to classify the feedbacks on " \
              "clinical efficacies from European Public Assessment Report (EPARs)"

gr.Interface(fn=__make_prediction__,
             inputs=[classifier, sentence],
             outputs="label",
             title=title,
             description=description) \
    .launch()
