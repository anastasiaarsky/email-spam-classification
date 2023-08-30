import argparse
import joblib
from sklearn.ensemble import RandomForestClassifier

from src.models import evaluation
from src.features import build_tfidf_features


def main():
    # Step 1: Get training flag (if applicable)
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('-t', '--train', action='store_true', help='Flag to indicate training')
    args = parser.parse_args()

    # Step 2: Get the training, validation, and testing sets
    X_train_vec, y_train, X_val_vec, y_val, X_test_vec, y_test = build_tfidf_features.tfidf()

    # Step 3: Train the RF model or load the pretrained RF model
    if args.train:
        print('\nModel training in progress...')
        # Step A: Create a default Random Forest Classifier since RandomizedSearchCV did not improve results
        rf = RandomForestClassifier()

        # Step B: Fit the RF Classifier on the training dataset
        rf.fit(X_train_vec, y_train)

        # Step C: Save the Random Forest model
        joblib.dump(rf, 'models/random_forest_model.joblib')
        print('Model training completed, trained model can be found at models/random_forest_model.joblib')

        # Step D: Evaluate on the validation dataset
        y_pred = rf.predict(X_val_vec)

        # Step E: Print metrics results
        print('Evaluation Results on the Validation Set:')
        evaluation.calculate_metrics(y_val, y_pred)

    else:
        print('\nModel prediction in progress...')
        # Step A: Load the pretrained model
        rf = joblib.load('models/random_forest_model.joblib')

        # Step B: Predict on the testing dataset
        y_pred = rf.predict(X_test_vec)
        print('Model prediction completed')

        # Step C: Print metrics results
        print('Random Forest Prediction Results on the Testing Set:')
        evaluation.calculate_metrics(y_test, y_pred)

        # Step D: Print confusion matrix
        evaluation.print_custom_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()
