from src.features import train_valid_test_split

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


def main(train=True):
    # Get the training, validation, and testing sets
    X_train, y_train, X_val, y_val, X_test, y_test = train_valid_test_split.main()

    # Word_level TF-IDF
    vec = TfidfVectorizer()

    # if training -> only fit the tf-idf vectorizer on the training data, then use it to transform the training dataset
    # and the validation dataset
    if train:
        vec.fit(X_train)
        X_train_vec = vec.transform(X_train)
        X_val_vec = vec.transform(X_val)
        print("Shape of Vectorized Training Set: {}, \nShape of Vectorized Validation Set: {}"
              .format(X_train_vec.shape, X_val_vec.shape))
        return X_train_vec, y_train, X_val_vec, y_val
    # if predicting -> fit the tf-idf vectorizer on the combined training and validation data (X_train_full), then use
    # it to transform the full training dataset and the testing dataset
    else:
        X_train_full = pd.concat([X_train, X_val])
        y_train_full = pd.concat([y_train, y_val])
        vec.fit(X_train_full)
        X_train_full_vec = vec.transform(X_train_full)
        X_test_vec = vec.transform(X_test)
        print("Shape of Vectorized Training Set: {}, \nShape of Vectorized Testing Set: {}"
              .format(X_train_full_vec.shape, X_test_vec.shape))
        return X_train_full_vec, y_train_full, X_test_vec, y_test


if __name__ == "__main__":
    main()
