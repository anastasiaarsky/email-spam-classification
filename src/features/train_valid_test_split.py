import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def main():
    # Load the processed data into a pandas dataframe and drop the (single) nan row
    df = pd.read_csv('data/processed_data.zip').dropna(subset='Text')

    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Label, test_size=0.3, shuffle=True, random_state=1)

    # Split the training dataset into training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=1)

    print("Length of training set:", len(X_train))
    print("Length of validation set:", len(X_val))
    print("Length of testing set:", len(X_test))

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    main()
