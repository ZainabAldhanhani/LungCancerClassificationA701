
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def Evaluation_reprot(y_test,y_pred,target_names=['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, target_names)
    # Print results
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

def Evaluation_confusion_matrix(y_test,y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma'],
                yticklabels=['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()