from src.features import train_valid_test_split
import os
import re
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense
from keras.layers import Dropout, Embedding, Flatten, GlobalMaxPooling1D, LSTM
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    # Get the training, validation, and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split.main()


if __name__ == "__main__":
    main()
