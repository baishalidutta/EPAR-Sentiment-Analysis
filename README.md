<p align="center">
  <img width="621" alt="logo" src="https://user-images.githubusercontent.com/76659596/113596851-2e5cc280-963b-11eb-8526-fb8fca9c837e.png">
</p>

<p align="center">
  <img width="110" alt="license" src="https://img.shields.io/badge/License-Apache-blue"/>
  <img width="100" alt="license" src="https://img.shields.io/badge/python-v3.7+-blue.svg"/>
</p>

<p align="center">
    <img width="1004" alt="webapp" src="https://user-images.githubusercontent.com/76659596/114297665-3771e780-9ab2-11eb-9238-d0f281e8e4cc.png">
</p>

## Motivation

Pharmaceutical companies need to submit scientific evidence and clinical trial data to support an application for a new drug or for an existing drug. The European Medicine Agency (EMA) is the central health authority 
in Europe and is responsible for the application process.

Following an application by a pharmaceutical company and an extensive scientific evaluation, the EMA publishes European Public Assessment Report (EPAR). EPARs are freely available and importantly contain the
scientific assessment of an application and reasons leading to refusal or approval.

Pharmaceutical companies extract relevant clinical efficacies from their EPARs that evaluate the setup and parameters of clinical trials. Regulatory colleagues from the pharmaceutical companies specify the sentiment
of the appropriate extractions (feedbacks), which can either be positive, negative or neutral.

This repository, therefore, strives to provide a tool to analyse the associated sentiments of the feedback above. Besides, it also provides a detailed comparison of various methods for sentiment analysis.

## Requirements

- Python 3.7.0+
- tensorflow 2.4.1+
- Keras 2.4.3+
- matplotlib 3.3.3+
- numpy 1.19.5+
- pandas 1.2.1+
- scikit-learn 0.24.1+ 
- nltk 3.5+
- spacy 3.0.3+
- textblob 0.15.3+
- gradio 1.5.3+
- click 7.1.2+
- xgboost 1.3.3+
- xlrd 2.0.1+
- seaborn 0.11.1+

## Dataset

The dataset cannot be provided since I don't have the copyright
for it.

But, you still can provide any XLSX dataset containing the following columns:

* `ID`
* `Sentence`
* `Positive`
* `Negative`
* `Neutral`

The `ID` is the index column and the `Positive`, `Negative` and `Neutral` columns
are one-hot encoded, that is, for every row, one of these columns contains `1` whereas 
the others contain `0`.

The following list enumerates different classes (types) of comments -

| Positive | Negative | Neutral |
|-------|------------|---------|

## Installation

* Clone the repository 

`git clone https://github.com/baishalidutta/EPAR-Sentiment-Analysis.git`

* Install the required libraries

`pip3 install -r requirements.txt`

## Model Classifiers

The dataset can be trained with several classifiers, including classical and deep learning models.

Currently, the following classical machine learning classifiers are implemented:

* `Naive Bayes`
* `Decision Tree`
* `Random Forest`
* `XGBoost`
* `Logistic Regression`
* `Linear Support Vector`
* `Kernel Support Vector Machine`

And, the following deep learning classifier:

* `Recurrent Neural Network` (RNN) with a `Bidirectional LSTM` layer

## Data Cleaning

* Lower all text
* Correct misspelled words
* Remove punctuations
* Remove stop words

## Data Preprocessing

* `TF-IDF Vectorizer` for classical machine learning models
* `Tokenize` text data and used Embedding Vector` using [Glove.6B](https://nlp.stanford.edu/projects/glove/) for deep learning model

## Usage

Navigate to the `source` directory to execute the following source code.

* To train or evaluate on your own, you can list the available options in the client application:

`python3 app.py --help`

<img width="566" alt="cli" src="https://user-images.githubusercontent.com/76659596/114533906-b3089b80-9c4e-11eb-8c81-22079cc06082.png">

To evaluate the analysis using default data (`../data/sentences_with_sentiment.xlsx)`, you can simply execute the following:

`python3 app.py --classifier ID`

This will execute the following in order (except deep learning model):

* `Train Test Validation` with default split of `0.2`
* `Cross-Validations`
    * `K-Folds Cross-Validation` with default `36` splits
    * `Leave One Out Cross-Validation`

You can also perform the `grid search` on classical machine learning classifiers:

`python3 app.py --grid`

This will execute the following in oder:

* `Grid Search` with `K-Folds Cross-Validation` with default `36` splits
* `Grid Search` with `Leave One Out Cross-Validation`

To execute the deep learning model, you can specify the associated ID, however, this will
not execute `K-Folds` or `Leave One Out` cross-validations.

Alternatively, you can find the complete analysis in the notebook inside the `notebook` directory. To open the notebook, use either `jupyter notebook` or `google colab` or any other IDE that supports notebook feature such as `PyCharm Professional`.

## Web Application

To run the web application locally, execute:

`python3 app_web.py`

This will start a local server that you can access in your browser. By default, the server will be started in `http://127.0.0.1:7860/`. You can type in any sentence, choose the classifier and find out which target class the classifier determines.

## Developer

Baishali Dutta (<a href='mailto:me@itsbaishali.com'>me@itsbaishali.com</a>)

## License [![License](http://img.shields.io/badge/license-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This project is licensed under Apache License Version 2.0
