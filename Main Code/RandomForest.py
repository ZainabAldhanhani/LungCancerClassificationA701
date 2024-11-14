from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from LoadExtractedFeatures import *
from EvaluatePerformance import *
import argparse

def test_best_radomforest_parameter(dataset_df,test_df,best_model_name):
    # Define the parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Set up Grid Search with 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='accuracy',  # You can also use 'f1_weighted' for F1 score
        cv=3,  # 3-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=2
    )

    X_train, y_train, X_test, y_test = load_extractedfeatures_train_and_test(dataset_df,test_df, best_model_name)
    # Run Grid Search on the training data
    grid_search.fit(X_train, y_train)

    cv_results = pd.DataFrame(grid_search.cv_results_)
    # Sort the results by mean test score in ascending order to find the worst-performing parameters
    worst_results = cv_results.sort_values(by="mean_test_score", ascending=True)

    # Display the worst parameter combinations and their scores
    print("\nWorst hyperparameters and their mean test scores:")
    expanded_cv_results = worst_results["params"].apply(pd.Series)
    expanded_cv_results["mean_test_score"] = cv_results["mean_test_score"]
    print(expanded_cv_results.head())

    # Display the worst parameter combinations and their scores
    print("\nBest hyperparameters and their mean test scores:")
    expanded_cv_results = worst_results["params"].apply(pd.Series)
    expanded_cv_results["mean_test_score"] = cv_results["mean_test_score"]
    print(expanded_cv_results.tail())

    print("***********************************************************")
    print("-----------------------------------------------------------")
    print("***********************************************************")

    # Get the best model with optimal hyperparameters
    best_rf_model = grid_search.best_estimator_
    print("Best hyperparameters:", grid_search.best_params_)
    return best_rf_model,X_train, y_train, X_test, y_test

# Evaluate the best model on the test set

def test_randomforest(dataset_df,test_df,best_model_name):
    best_rf_model,X_train, y_train, X_test, y_test=test_best_radomforest_parameter(dataset_df,test_df,best_model_name)
    y_pred = best_rf_model.predict(X_test)
    Evaluation_report(y_test,y_pred)
    Evaluation_confusion_matrix(y_test,y_pred)
    
    
def main(train_dataset_path, test_dataset_path,best_model_name):
    dataset_df = get_profile_path(train_dataset_path)
    test_df = get_profile_path(test_dataset_path)
    test_randomforest(dataset_df,test_df,best_model_name)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classification Using Random Forest.")
    
    # Define parameters
    parser.add_argument("--train_dataset_path", type=str, required=True, help="Put your train dataset path")
    parser.add_argument("--test_dataset_path", type=str, required=True, help="Put your test dataset path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")

    # Parse arguments
    args = parser.parse_args()
    main(args.train_dataset_path, args.test_dataset_path, args.model_name)