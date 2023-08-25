from src.features import build_ml_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Function to print metrics results
def calculate_metrics(y_true, y_pred):
    print("Random Forest, Word Level TF-IDF Results:")
    print("Test accuracy: {:.3f}".format(accuracy_score(y_true, y_pred) * 100))
    print("F1 Score: {:.3f}".format(f1_score(y_true, y_pred, average='macro') * 100))
    print("Recall: {:.3f}".format(recall_score(y_true, y_pred, average='macro') * 100))
    print("Precision: {:.3f}".format(precision_score(y_true, y_pred, average='macro') * 100))


def main():
    # Get the training, validation, and testing sets
    X_train_vec, y_train, X_val_vec, y_val = build_ml_features.main()

    # Random Forest Training Results on Word Level TF-IDF Vectors
    # (using the training and validation sets)
    #
    # Use the default Random Forest Classifier since RandomizedSearchCV
    # produced worse results compared to the default
    classifier = RandomForestClassifier()

    # Fit the training dataset on the classifier
    classifier.fit(X_train_vec, y_train)

    # Predict the labels on validation dataset
    y_pred = classifier.predict(X_val_vec)

    # Print metrics results
    calculate_metrics(y_val, y_pred)


if __name__ == "__main__":
    main()
