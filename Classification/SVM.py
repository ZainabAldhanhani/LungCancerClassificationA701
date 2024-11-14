from sklearn.svm import SVC
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from EvaluatePerformance import *
from LoadExtractedFeatures import *
import argparse
# Function to train and evaluate SVMs with different hyperparameters
def train_and_evaluate_svm_kfold(features, labels,kernels,C_values, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_model = None
    best_f1_score = 0
    results = []
    if labels is not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # List of kernels and hyperparameters to try

    # Iterate over all combinations of kernels and C values
    for kernel in kernels:
        for C in C_values:
            accuracies = []
            f1_scores = []

            for train_idx, val_idx in kf.split(features):
                X_train, X_val = features[train_idx], features[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]

                # Train the SVM
                svm = SVC(kernel=kernel, C=C)
                svm.fit(X_train, y_train)

                # Validate the SVM
                val_preds = svm.predict(X_val)
                accuracy = accuracy_score(y_val, val_preds)
                f1 = f1_score(y_val, val_preds, average='weighted')

                accuracies.append(accuracy)
                f1_scores.append(f1)

            mean_accuracy = np.mean(accuracies)
            mean_f1 = np.mean(f1_scores)
            results.append((kernel, C, mean_accuracy, mean_f1))

            # Save the best model based on F1 score
            if mean_f1 > best_f1_score:
                best_f1_score = mean_f1
                best_model = svm
                best_kernel = kernel
                best_C = C
    # Print summary of all results
    print("\nAll results:")
    for kernel, C, mean_accuracy, mean_f1 in results:
        print(f"Kernel: {kernel}, C: {C} - Mean Accuracy: {mean_accuracy:.4f}, Mean F1 Score: {mean_f1:.4f}")
    
    # Save the best model to a file
    if best_model:
        joblib.dump(best_model, 'best_svm_model.pkl')
        print(f"\nBest model saved with F1 Score: {best_f1_score:.4f} (Kernel: {best_kernel}, C: {best_C})")
    else:
        print("No model found.")

    
    return best_kernel,best_C

def train_best_svm_on_full_data(features, labels, best_kernel, best_C):
    # Ensure labels are a NumPy array
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Train the SVM with the best kernel and C on the entire dataset
    print(f"\nTraining the best SVM with kernel={best_kernel} and C={best_C} on the full training set.")
    best_svm_model = SVC(kernel=best_kernel, C=best_C)
    best_svm_model.fit(features, labels)

    # Save the trained model
    joblib.dump(best_svm_model, 'best_svm_model_full_data.pkl')
    print("\nBest SVM model trained on the full training set and saved as 'best_svm_model_full_data.pkl'.")

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def test_best_svm_classifier(test_features, test_labels):
    # Load the best SVM model
    try:
        best_svm_model = joblib.load('best_svm_model_full_data.pkl')
        print("Best SVM model loaded successfully.")
    except FileNotFoundError:
        print("Error: Best SVM model file not found.")
        return

    # Ensure test labels are a NumPy array
    if not isinstance(test_labels, np.ndarray):
        test_labels = np.array(test_labels)

    # Predict using the loaded model
    test_preds = best_svm_model.predict(test_features)

    # Evaluate performance
    Evaluation_report(test_labels,test_preds)
    Evaluation_confusion_matrix(test_labels,test_preds)

def main(train_dataset_path, test_dataset_path,best_model_name):
    dataset_df = get_profile_path(train_dataset_path)
    test_df = get_profile_path(test_dataset_path)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_values = [0.001, 0.01, 1.0, 10.0, 100.0]
    best_model_name='densenet121'
    X_train, y_train, X_test, y_test = load_extractedfeatures_train_and_test(dataset_df,test_df, best_model_name)
    best_kernel,best_C = train_and_evaluate_svm_kfold(X_train, y_train,kernels,C_values,n_splits=5)
    train_best_svm_on_full_data(X_train, y_train, best_kernel = best_kernel, best_C = best_C)
    test_best_svm_classifier(X_test, y_test)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classification Using SVM")
    
    # Define parameters
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Put your train dataset path")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Put your test dataset path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    # Parse arguments
    args = parser.parse_args()
    main(args.train_dataset_path, args.test_dataset_path, args.model_name)