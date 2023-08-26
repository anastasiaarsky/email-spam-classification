import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def main(train=True):
    # Load the processed data into a pandas dataframe and drop the (single) nan row
    df = pd.read_csv('data/processed_data.zip').dropna(subset='Text')

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Label, test_size=0.3, shuffle=True, random_state=1)

    # Word_level TF-IDF
    vec = TfidfVectorizer()

    # if training -> only fit the tf-idf vectorizer on the training data, then use it to transform the training dataset
    # and the validation dataset
    if train:
        # Fit the vectorizer on the training data
        vec.fit(X_train)
        # Split the training dataset into training and validation datasets and do tf-idf transformation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=1)
        X_train_vec = vec.transform(X_train)
        X_val_vec = vec.transform(X_val)
        print("Shape of Vectorized Training Set: {}, \nShape of Vectorized Validation Set: {}"
              .format(X_train_vec.shape, X_val_vec.shape))
        return X_train_vec, y_train, X_val_vec, y_val
    # if predicting -> fit the tf-idf vectorizer on the combined training and validation data (X_train_full), then use
    # it to transform the full training dataset and the testing dataset
    else:
        vec.fit(X_train)
        X_train_full_vec = vec.transform(X_train)
        X_test_vec = vec.transform(X_test)
        print("Shape of Vectorized Training Set: {}, \nShape of Vectorized Testing Set: {}"
              .format(X_train_full_vec.shape, X_test_vec.shape))
        return X_train_full_vec, y_train, X_test_vec, y_test


if __name__ == "__main__":
    main()
