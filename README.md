<p align="center">
  <img width="621" alt="logo" src="https://user-images.githubusercontent.com/76659596/113596851-2e5cc280-963b-11eb-8526-fb8fca9c837e.png">
</p>

<p align="center">
  <img width="110" alt="license" src="https://img.shields.io/badge/License-Apache-blue"/>
  <img width="100" alt="license" src="https://img.shields.io/badge/python-v3.7+-blue.svg"/>
</p>

<p align="center">
    <img width="700" alt="webapp" src="https://user-images.githubusercontent.com/76659596/113773171-f2525c00-9725-11eb-964a-4aa7231dda78.png">
</p>

## Motivation

Pharmaceutical companies need to submit scientific evidence and clinical trial data to support an application for a new drug or for an existing drug. The European Medicine Agency (EMA) is the central health authority 
in Europe and is responsible for the application process.

Following an application by a pharmaceutical company and an extensive scientific evaluation, the EMA publishes a European Public Assessment Report (EPARs). EPARs are freely available and importantly contain the
scientific assessment of an application and reasons leading to refusal or approval.

Pharmaceutical companies extract relevant clinical efficacies from their EPARs that evaluate the setup and parameters of clinical trials. Regulatory colleagues from the pharmaceutical companies specify the sentiment
of the appropriate extractions (feedbacks), which can either be positive, negative or neutral.

This repository, therefore, strives to provide a tool to analyse the associated sentiments of the feedback above. Besides, it also provides a detailed comparison of various methods for sentiment
analysis.

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

I cannot provide the dataset since I don't have the copyright for it.

But, you still can provide an XLSX dataset containing the following columns:

* ID
* Sentence
* Positive
* Negative
* Neutral

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

* Naive Bayes
* Decision Tree
* Random Forest
* XGBoost
* Logistic Regression
* Linear Support Vector
* Kernel Support Vector Machine

And, the following deep learning classifier:

* Recurrent Neural Network (RNN) with a `Bidirectional LSTM` layer

## Data Cleaning Steps

* lower all text
* correct misspeled words
* remove punctuations
* remove stop words

## Deep Learning Model Ideology

* `Tokenize text` data
* Create `Embedding Vector` using [Glove.6B](https://nlp.stanford.edu/projects/glove/)
* Train the deep learning model of your choice

## Usage

Navigate to the `source` directory to execute the following source code.

* To train or evaluate on your own, execute the following:

<img width="500" alt="cli" src="https://user-images.githubusercontent.com/76659596/113787147-c5f50a80-973a-11eb-8aa9-dff53630cf3e.png">

Alternatively, you can find the complete analysis in the notebook inside the `notebook` directory. To open the notebook, use either `jupyter notebook` or `google colab` or any other IDE that supports notebook feature such as `PyCharm Professional`.

## Web Application

To run the web application locally, execute:

`python3 app_web.py`

This will start a local server that you can access in your browser. By default, the server will be started in `http://127.0.0.1:7860/`. You can type in any sentence, choose the classifier and find out what polarity the classifier determines.

## Developer

Baishali Dutta (<a href='mailto:me@itsbaishali.com'>me@itsbaishali.com</a>)

## License [![License](http://img.shields.io/badge/license-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This project is licensed under Apache License Version 2.0
