
from crossValidateTrain import *
from dataset import *
from loadImagepaths import get_profile_path
from RandomForest import *
from SVM import *
from testDatasetonEachModel import *
from visualization import *

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    size = 224
    dataset_df = get_profile_path('/Users/zainabaldhanhani/Desktop/AI701Project/train')
    test_df = get_profile_path('/Users/zainabaldhanhani/Desktop/AI701Project/test')

    print(f'Total train images loaded: {len(dataset_df)}')# Check if dataset_df is not empty
    print(f'Total test images loaded: {len(test_df)}')# Check if dataset_df is not empty
    # Call the visualization function to view dataset samples
    visualize_dataset(dataset_df)

    # List of models to test
    model_names = ['vgg16', 'resnet50', 'densenet121']
    model_weights = {
        'vgg16': 'VGG16_Weights.DEFAULT',
        'resnet50': 'ResNet50_Weights.DEFAULT',
        'densenet121': 'DenseNet121_Weights.DEFAULT'
    }

    #######################################
    #Search for the best modle 
    # Train and evaluate all models, get the best model and its name
    best_model, best_model_name, results = cross_validate_and_train(dataset_df, model_names,model_weights, size, device,)

    # Print all model results
    print("\nAll Model Results:")
    for model, scores in results.items():
        print(f"{model} - Accuracy: {scores['accuracy']:.4f}, F1-Score: {scores['f1_score']:.4f}")

    #######################################


    # Test the models on the test set and save the best model
    test_results, best_model_name, best_f1_score = test_on_test_set_and_save_best(test_df, model_names, model_weights, size, device, 'best_model.pth')

    test_randomforest(dataset_df,test_df,best_model_name)


    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_values = [0.001, 0.01, 1.0, 10.0, 100.0]
    best_model_name='densenet121'
    X_train, y_train, X_test, y_test = load_extractedfeatures_train_and_test(dataset_df,test_df, best_model_name)
    best_kernel,best_C = train_and_evaluate_svm_kfold(X_train, y_train,kernels,C_values,n_splits=5)
    train_best_svm_on_full_data(X_train, y_train, best_kernel = best_kernel, best_C = best_C)
    test_best_svm_classifier(X_test, y_test)