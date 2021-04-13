__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import pickle

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Dropout, GlobalMaxPooling1D, LSTM, Bidirectional, Embedding
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import roc_auc_score, accuracy_score

from source.classifier.classifier import Classifier
from source.util.data_preprocessing import clean_text, get_dataset

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIMENSION = 100
MAX_SEQUENCE_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 30
PATIENCE = 5
DETECTION_CLASSES = ['Positive', 'Negative', 'Neutral']
MODEL_LOC = "../model/epar_sentiment_analysis_lstm.h5"
PLOT_LOC = "../plots/"
EMBEDDING_FILE_LOC = '../model/glove/glove.6B.' + str(EMBEDDING_DIMENSION) + 'd.txt'
TOKENIZER_LOC = '../model/tokenizer.pickle'
SENTENCE_COLUMN = "Sentence"


class LstmClassifier(Classifier):
    """
    The Long Short Term Memory Recurrent Neural Network (RNN) model
    for training and evaluation
    """

    def __init__(self):
        self.name = ["Bidirectional Long Short Term Memory (RNN)"]

    def evaluate(self, data, testsplit, cvsplit):
        """
        The evaluation operation to perform. It will
        train the model before evaluation.

        :param data: the data to perform the evaluation on
        :param testsplit: the test split on the data
        :param cvsplit: the cross-validation split on the data (not used for LSTM)
        """
        preprocessing = self.DataPreprocess(get_dataset(data))
        rnn_model, history = self.__build_rnn_model__(preprocessing.padded_data,
                                                      preprocessing.target_classes,
                                                      preprocessing.embedding_layer,
                                                      testsplit)

        self.__plot_training_history__(rnn_model,
                                       history,
                                       preprocessing.padded_data,
                                       preprocessing.target_classes)

    def predict(self, data, testsplit, cvsplit, input_sentence):
        """
        The prediction on single input sentence

        :param data: the data to train on before prediction
                        (not required for LSTM since the model will
                        be available before the prediction, hence, we don't
                        need to train the model while performing prediction)
        :param testsplit: the test split for training
                        (not required for LSTM since the model will
                        be available before the prediction and we don't
                        need to perform any test split for training while
                        predicting)
        :param cvsplit: the cross-validation split on the data (not used for LSTM)
        :param input_sentence: the single sentence to predict the classification on
        :return: the target prediction class
        """
        # load the trained model
        rnn_model = load_model(MODEL_LOC)

        # load the tokenizer
        with open(TOKENIZER_LOC, 'rb') as handle:
            tokenizer = pickle.load(handle)

        input_sentence = clean_text(input_sentence)
        input_sentence = input_sentence.split(" ")

        sequences = tokenizer.texts_to_sequences(input_sentence)
        sequences = [[item for sublist in sequences for item in sublist]]

        padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return rnn_model.predict(padded_data, len(padded_data), verbose=1)

    def __build_rnn_model__(self, data, target_classes, embedding_layer, split):
        """
        Build and Train the RNN architecture (Bidirectional LSTM)

        :param data: the preprocessed padded data
        :param target_classes: assigned target labels for the sentences
        :param embedding_layer: embedding layer comprising preprocessed sentences
        :param split: the validation split for training
        :return: the trained model and the history
        """
        # Create an LSTM Network with a single LSTM
        input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
        x = embedding_layer(input_)
        x = Bidirectional(LSTM(units=64,
                               return_sequences=True,
                               recurrent_dropout=0.2))(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(units=64, activation='relu')(x)
        x = Dropout(rate=0.2)(x)
        # x = GlobalMaxPooling1D()(x)

        #  Sigmoid Classifier
        output = Dense(len(DETECTION_CLASSES), activation="sigmoid")(x)

        model = Model(input_, output)

        # Display Model
        model.summary()

        # Compile Model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Define Callbacks
        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=PATIENCE,
                                   mode='min',
                                   restore_best_weights=True)

        checkpoint = ModelCheckpoint(filepath=MODEL_LOC,  # saves the 'best' model
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min')

        # Fit Model
        history = model.fit(data,
                            target_classes,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_split=split,
                            callbacks=[early_stop, checkpoint],
                            verbose=1)

        # Return Model Training History
        return model, history

    def __plot_training_history__(self, rnn_model, history, data, target_classes):
        """
        Generates plots for accuracy and loss

        :param rnn_model: the trained model
        :param history: the model history
        :param data: preprocessed data
        :param target_classes: target classes for every sentence
        """
        #  "Accuracy"
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(PLOT_LOC + "accuracy-lstm.jpeg")
        plt.show()

        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(PLOT_LOC + "loss-lstm.jpeg")
        plt.show()

        # Print Average ROC_AUC_Score
        p = rnn_model.predict(data)
        aucs = []
        for j in range(len(DETECTION_CLASSES)):
            auc = roc_auc_score(target_classes[:, j], p[:, j])
            aucs.append(auc)
        print("======================================================")
        print("                     ROC_AUC Score                    ")
        print("======================================================")
        print(f'{" ":<17} {np.mean(aucs)}')

    def __make_prediction__(self, preprocessing):
        """
        Performs prediction on the padded data from the preprocessing instance

        :param preprocessing: prepared DataPreprocess instance
        :return: the loaded model instance
        """
        rnn_model = load_model(MODEL_LOC)
        prediction = rnn_model.predict(preprocessing.padded_data,
                                       steps=len(preprocessing.padded_data) / BATCH_SIZE,
                                       verbose=1)
        return prediction

    def __evaluate_roc_auc__(self, preprocessing, prediction_binary):
        """
        Evaluates the model

        :param preprocessing: prepared DataPreprocess instance
        :param prediction_binary: boolean expression for the predicted classes
        """
        aucs = []
        for j in range(len(DETECTION_CLASSES)):
            auc = roc_auc_score(preprocessing.target_classes[:, j], prediction_binary[:, j])
            aucs.append(auc)

        return np.mean(aucs)

    def __evaluate_accuracy_score__(self, preprocessing, prediction_binary):
        """
        Evaluates the accuracy score

        :param preprocessing: prepared DataPreprocess instance
        :param prediction_binary: boolean expression for the predicted classes
        """
        accuracy = []
        for j in range(len(DETECTION_CLASSES)):
            acc = accuracy_score(preprocessing.target_classes[:, j], prediction_binary[:, j])
            accuracy.append(acc)

        return np.mean(accuracy)

    class DataPreprocess:
        """
        Preprocesses the data for LSTM
        """

        def __init__(self, data, do_load_existing_tokenizer=False):
            """
            Initializes and prepares the data with necessary steps either to be trained
            or evaluated by the RNN model

            :param data: the dataframe extracted from the .csv file
            :param do_load_existing_tokenizer: True if existing tokenizer should be loaded or False instead
            """
            self.data = data
            self.doLoadExistingTokenizer = do_load_existing_tokenizer

            # The pre-trained word vectors used (http://nlp.stanford.edu/data/glove.6B.zip)
            word_to_vector = {}
            with open(EMBEDDING_FILE_LOC) as file:
                # A space-separated text file in the format
                # word vec[0] vec[1] vec[2] ...
                for line in file:
                    word = line.split()[0]
                    word_vec = line.split()[1:]
                    # converting word_vec into numpy array
                    # adding it in the word_to_vector dictionary
                    word_to_vector[word] = np.asarray(word_vec, dtype='float32')

            # Print the total words found
            print("======================================================")
            print("                 Total Word Vectors                   ")
            print("======================================================")
            print(f'{" ":<21} {len(word_to_vector)}')

            # Clean the sentences
            data[SENTENCE_COLUMN] = data[SENTENCE_COLUMN].apply(lambda x: clean_text(x))

            # Split the data into feature and target labels
            sentences = data[SENTENCE_COLUMN].values
            self.target_classes = data[DETECTION_CLASSES].values

            if not do_load_existing_tokenizer:
                # Convert the sentences (strings) into integers
                tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
                tokenizer.fit_on_texts(sentences)
            else:
                with open(TOKENIZER_LOC, 'rb') as handle:
                    tokenizer = pickle.load(handle)

            sequences = tokenizer.texts_to_sequences(sentences)

            # Word to integer mapping
            word_to_index = tokenizer.word_index
            print("======================================================")
            print("                 Total Unique Tokens                  ")
            print("======================================================")
            print(f'{" ":<22} {len(word_to_index)}')

            if not do_load_existing_tokenizer:
                # Save tokenizer
                with open(TOKENIZER_LOC, 'wb') as handle:
                    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Pad sequences so that we get a N x T matrix
            self.padded_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            print("======================================================")
            print("                 Shape of Data Tensor                 ")
            print("======================================================")
            print(f'{" ":<19} {self.padded_data.shape}')

            # Construct and Prepare Embedding matrix
            num_words = min(MAX_VOCAB_SIZE, len(word_to_index) + 1)
            embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSION))
            for word, i in word_to_index.items():
                if i < MAX_VOCAB_SIZE:
                    embedding_vector = word_to_vector.get(word)
                    if embedding_vector is not None:
                        # words not found in embedding index will be all zeros.
                        embedding_matrix[i] = embedding_vector

            # Load pre-trained word embeddings into an embedding layer
            # Set trainable = False to keep the embeddings fixed
            self.embedding_layer = Embedding(num_words,
                                             EMBEDDING_DIMENSION,
                                             weights=[embedding_matrix],
                                             input_length=MAX_SEQUENCE_LENGTH,
                                             trainable=False)
