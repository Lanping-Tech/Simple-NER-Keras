import numpy as np

import keras
from keras.models import Sequential, model_from_json, load_model
from keras.layers import *
from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from data_preprocessing import *
from metrics import *


WORD_EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 15

train_path = './data/eng.train'
test_path = './data/eng.testa'

num_classes = 5

vocab = build_vocab([train_path, test_path])
num_words = len(vocab)


model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=WORD_EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Conv1D(filters=100, kernel_size=3, padding='same', activation='relu'))
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[precision, recall, f1])

x_train, y_train = load_data(train_path, vocab, MAX_SEQUENCE_LENGTH)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_test, y_test = load_data(test_path, vocab, MAX_SEQUENCE_LENGTH)
y_test = keras.utils.to_categorical(y_test, num_classes)

model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_test, y_test))
