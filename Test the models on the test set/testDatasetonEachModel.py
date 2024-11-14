from dataset import *
from loadImagepaths import *
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Function to test each model on a separate test set and save the best model
def test_on_test_set_and_save_best(test_df, model_names, model_weights, size, device, save_path="best_model.pth"):
    test_results = {}
    best_f1_score = 0
    best_model_name = ""
    best_model_state = None

    for model_name in model_names:
        print(f"\nTesting {model_name} on the test set")

        # Load the model with pretrained weights and move to the device
        model = getattr(torchvision.models, model_name)(weights=model_weights[model_name]).to(device)
        model.load_state_dict(torch.load(f'{model_name}_weights.pth', map_location=torch.device("cuda" if torch.cuda.is_available() else 'cpu')), strict=False)
        model.eval()

        # Set up test dataset and loader
        test_dataset = Dataset(test_df, size)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)

        # Evaluate the model on the test set
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        # Calculate metrics
        test_accuracy = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='weighted')

        test_results[model_name] = {'test_accuracy': test_accuracy, 'test_f1_score': test_f1}
        print(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Test F1-Score: {test_f1:.4f}")

        # Check if this model is the best so far based on F1-score
        if test_f1 > best_f1_score:
            best_f1_score = test_f1
            best_model_name = model_name
            best_model_state = model.state_dict()  # Save the state dict of the best model

    # Save the best model's state dict
    if best_model_state:
        torch.save(best_model_state, save_path)
        print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1_score:.4f} saved to {save_path}")

    return test_results, best_model_name, best_f1_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    size = 224
    dataset_df = get_profile_path('/Users/zainabaldhanhani/Desktop/AI701Project/train')
    test_df = get_profile_path('/Users/zainabaldhanhani/Desktop/AI701Project/test')

    print(f'Total train images loaded: {len(dataset_df)}')# Check if dataset_df is not empty
    print(f'Total test images loaded: {len(test_df)}')# Check if dataset_df is not empty
    # Call the visualization function to view dataset samples
    #visualize_dataset(dataset_df)

    # List of models to test
    model_names = ['vgg16', 'resnet50', 'densenet121']
    model_weights = {
        'vgg16': 'VGG16_Weights.DEFAULT',
        'resnet50': 'ResNet50_Weights.DEFAULT',
        'densenet121': 'DenseNet121_Weights.DEFAULT'
    }

    test_results, best_model_name, best_f1_score = test_on_test_set_and_save_best(test_df, model_names, model_weights, size, device, 'best_model.pth')


