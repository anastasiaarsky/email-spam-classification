import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense
from keras.layers import Dropout, Embedding, Flatten, GlobalMaxPooling1D, LSTM
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier