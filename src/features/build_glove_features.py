import zipfile
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


# Input: training and testing sets (pandas dfs), maximum length of vectorized sequences
# Output: vectorized and padded training and testing sets (pandas dfs), dictionary that maps each unique word in the
# training set to a unique index
def vectorize_data(X_train, X_test, max_len):
    # Get number of unique words in the training set
    unique_words = set()
    X_train.str.lower().str.split(" ").apply(unique_words.update)
    vocab_size = len(unique_words)

    # Fit the tokenizer onto the training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="")
    tokenizer.fit_on_texts(X_train)

    # Convert the training and testing sets to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences so they are all the same length (500)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

    return X_train_pad, X_test_pad, tokenizer.word_index


# Input: dictionary of unique words and indexes (from the training set), num unique words, size of embeddings (300)
# Output: an embedding matrix where each unique word in the training set is represented by a vector of size 300
def load_pretrained_embeddings(word_to_index, vocab_size, embed_size, path_to_embeddings):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    with zipfile.ZipFile(path_to_embeddings, 'r') as z:
        embeddings_index = dict(get_coefs(*row.decode('utf8').split(" "))
                                for row in z.open('wiki-news-300d-1M-subword.vec')
                                if len(row) > 100)

    all_embs = np.stack(list(embeddings_index.values()))
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, embed_size))

    for word, idx in word_to_index.items():
        if idx >= vocab_size:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector

    return embedding_matrix


def glove(predict=False):
    # Step 1: Load the processed data into a pandas dataframe and drop the (single) nan row
    df = pd.read_csv('data/processed_data.zip').dropna(subset='Text')

    # Step 2: Split the data into training and testing datasets (85:15 to follow 70:15:15 rule)
    X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Label, test_size=0.15, shuffle=True, random_state=1)

    # Step 3: Vectorize training and testing datasets
    # max len is the maximum length of vectorized sequences
    max_len = 500
    # call vectorize_data and get the vectorized training and testing sets, along with a word_to_index dict for the
    # training set
    X_train_vec, X_test_vec, word_to_index = vectorize_data(X_train, X_test, max_len)

    # if simply predicting, stop here and return the vectorized testing set
    if predict:
        print('Vectorization on testing set completed')
        return X_test_vec, y_test

    # if training, continue to build the features
    else:
        print('Vectorization on training and validation sets completed')
        # Step 4: Split the training dataset into training and validation datasets (80:20 to follow 70:15:15 rule)
        X_train_vec, X_val_vec, y_train, y_val = train_test_split(X_train_vec, y_train, test_size=0.2, shuffle=True,
                                                                  random_state=1)
        # print('\tNumber of training samples: {:,}'.format(X_train_vec.shape[0]))
        # print('\tNumber of validation samples: {:,}'.format(X_val_vec.shape[0]))
        # print('\tNumber of testing samples: {:,}'.format(X_test_vec.shape[0]))

        # Step 5: Find embedding for every word in the training and validation datasets
        # vocab_size is the number of unique words in the training and validation sets
        vocab_size = len(word_to_index) + 1
        # print('\tSize of vocabulary (unique words in the corpus): {:,}'.format(vocab_size))
        # embed_size is the size of embeddings (i.e. the length of the vectors that represent each unique word)
        embed_size = 300
        path_to_embeddings = 'data/external_data/wiki-news-300d-1M-subword.vec.zip'
        # call load_pretrained_embeddings and get an embedding matrix, where each unique word in the training &
        # validation sets is represented by a vector
        ft_embeddings = load_pretrained_embeddings(word_to_index, vocab_size, embed_size, path_to_embeddings)

        # Step 6: Create the embedding layer
        embedding_layer = Embedding(input_dim=vocab_size,
                                    output_dim=embed_size,
                                    input_length=max_len,
                                    weights=[ft_embeddings],
                                    trainable=True)
        print('FastText embedding layer created')

        return X_train_vec, y_train, X_val_vec, y_val, embedding_layer


if __name__ == "__main__":
    glove()
