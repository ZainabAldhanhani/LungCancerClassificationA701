from sklearn.model_selection import KFold
import torchvision
from dataset import *
from trainEvaluate import *
# Cross-validation and model evaluation

def cross_validate_and_train(df, model_names, model_weights,size, device,cpu=True, n_splits=5):
    best_model = None
    best_model_name = ""
    best_f1_score = 0
    results = {}

    for model_name in model_names:
        print(f"\nTesting model: {model_name}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        f1_scores = []

        for train_idx, val_idx in kf.split(df):
            train_data = df.iloc[train_idx]
            val_data = df.iloc[val_idx]

            # Load model
            model = getattr(torchvision.models, model_name)(weights=model_weights[model_name]).to(device)

            # Datasets and loaders
            train_dataset = Dataset(train_data, size)
            val_dataset = Dataset(val_data, size)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2 if torch.cuda.is_available() else 0)

            # Train and evaluate
            train_and_evaluate(model_name, model, train_loader, val_loader, accuracies, f1_scores, device)

        mean_f1 = np.mean(f1_scores)
        mean_accuracy = np.mean(accuracies)
        results[model_name] = {'accuracy': mean_accuracy, 'f1_score': mean_f1}

        print(f"{model_name} - Mean F1-Score: {mean_f1:.4f}, Mean Accuracy: {mean_accuracy:.4f}")

        # Track the best model based on F1-score
        if mean_f1 > best_f1_score:
            best_f1_score = mean_f1
            best_model = model
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1_score:.4f}")
    return best_model, best_model_name, results
