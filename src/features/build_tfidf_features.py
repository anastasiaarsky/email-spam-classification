import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf():
    # Step 1: Load the processed data into a pandas dataframe and drop the (single) nan row
    df = pd.read_csv('data/processed_data.zip').dropna(subset='Text')

    # Step 2: Create the word_level TF-IDF vectorizer
    vec = TfidfVectorizer()

    # Step 3 & 4: Split the data using the 70-15-15 rule (70% training : 15% validation : 15% testing) and
    # fit the vectorizer on the training (and validation) data
    #
    # first, split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Label, test_size=0.15, shuffle=True,
                                                        random_state=1)
    # fit the vectorizer on the training data
    vec.fit(X_train)
    # split the training dataset into training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=1)

    # Step 5: Transform the training, validation, and testing datasets using the fitted vectorizer
    X_train_vec = vec.transform(X_train)
    X_val_vec = vec.transform(X_val)
    X_test_vec = vec.transform(X_test)
    print('TF-IDF vectorization on training, validation, and testing sets completed')
    # print('\tNumber of training samples: {:,}'.format(X_train_vec.shape[0]))
    # print('\tNumber of validation samples: {:,}'.format(X_val_vec.shape[0]))
    # print('\tNumber of testing samples: {:,}'.format(X_test_vec.shape[0]))
    # print('\tNumber of features (unique words in the corpus): {:,}'.format(X_train_vec.shape[1]))

    return X_train_vec, y_train, X_val_vec, y_val, X_test_vec, y_test


if __name__ == "__main__":
    tfidf()
