from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


# Function to print metrics results
def calculate_metrics(y_true, y_pred):
    print('\tTest accuracy: {:.3f}'.format(accuracy_score(y_true, y_pred) * 100))
    print('\tF1 Score: {:.3f}'.format(f1_score(y_true, y_pred, average='macro') * 100))
    print('\tRecall: {:.3f}'.format(recall_score(y_true, y_pred, average='macro') * 100))
    print('\tPrecision: {:.3f}'.format(precision_score(y_true, y_pred, average='macro') * 100))


# Function to print a custom confusion matrix with added labels
def print_custom_confusion_matrix(y_true, y_pred):
    labels = ['True Negative', 'Predicted Positive', 'Actual Negative', 'Actual Positive']
    cm = confusion_matrix(y_true, y_pred)

    print('\tConfusion Matrix:')
    print('\t{:20} {:^20} {:^20}'.format("", labels[0], labels[1]))
    print('\t{:20} {:^20} {:^20}'.format(labels[2], cm[0, 0], cm[0, 1]))
    print('\t{:20} {:^20} {:^20}'.format(labels[3], cm[1, 0], cm[1, 1]))