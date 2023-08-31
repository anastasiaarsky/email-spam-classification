import argparse
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, GlobalMaxPooling1D
from keras.models import Sequential, load_model

from src.features import build_glove_features
from src.models import evaluation


def main():
    # Step 1: Get training flag (if applicable)
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-t', '--train', action='store_true', help='Flag to indicate training')
    args = parser.parse_args()

    # Step 2: Train the CNN model or load the pretrained CNN model to make predictions
    if args.train:
        # Step A: Get the training and validation sets, as well as the embedding layer
        X_train_vec, y_train, X_val_vec, y_val, embedding_layer = build_glove_features.glove()

        print('\nModel training in progress...')
        # Step B: Create a CNN model that uses the FastText word embeddings
        cnn = Sequential([
            embedding_layer,
            Conv1D(128, 5, padding='same', activation='relu'),
            GlobalMaxPooling1D(),
            Dense(10, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Step C: Compile the CNN
        cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Step D: Train the CNN for 10 epochs with early stopping (patience of 2)
        earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
        cnn.fit(X_train_vec, y_train, epochs=10,
                callbacks=[earlystop],
                validation_data=(X_val_vec, y_val))

        # Step E: Save the CNN model
        cnn.save('models/cnn_model.h5')
        print('Model training completed, trained model can be found at models/cnn_model.h5')

    else:
        # Step A: Get the testing set
        X_test_vec, y_test = build_glove_features.glove(predict=True)

        print('\nModel prediction in progress...')
        # Step B: Load the pretrained model
        cnn = load_model('models/cnn_model.h5')

        # Step C: Predict on the testing dataset
        y_pred = (cnn.predict(X_test_vec) > 0.5).astype("int32")
        print('Model prediction completed')

        # Step 5: Print metrics results
        print('CNN Prediction Results on the Testing Set:')
        evaluation.calculate_metrics(y_test, y_pred)

        # Step 6: Print confusion matrix
        evaluation.print_custom_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()
