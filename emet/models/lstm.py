import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, \
                         Dropout, Input, Bidirectional, \
                         TimeDistributed
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

dataset = pd.read_pickle('./dataset/multilingual_concatenated_sentences.pkl')

X = []

for sentence in dataset.concatenated_sentences:

    X.append(sentence.reshape((1024,)))

X = np.asarray(X)

Y = dataset['encoded_label'].values

X_train, X_test, y_train, y_test = train_test_split(X, \
                                                    Y, \
                                                    test_size = 0.2, \
                                                    random_state = 42)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
y_train = y_train.reshape((y_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_test = y_test.reshape((y_test.shape[0], 1, 1))

def __build_model():

    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(1, 1024)))
    # model.add(LSTM(64, return_sequences=True, input_shape=(1, 1024), activation='sigmoid'))
    model.add(TimeDistributed(Dense(20, activation='sigmoid')))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, activation='relu'))  # return a single vector of dimension 32)
    # model.add(Dense(1, activation='relu'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def fit_model(X, y):

    model = __build_model()

    model.fit(X, y, batch_size=50, epochs=20, \
              validation_split=0.2, \
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

fit_model(X_train, y_train)
