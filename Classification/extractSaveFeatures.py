# Extract and save features as .pth file
from dataset import *
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set device and image size

def extract_and_save_features(df, filename, model,size=244):
    model.eval()
    dataset = Dataset(df, size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    all_features = []
    all_labels = []  # Optional: save labels as well if needed


    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)  # Move inputs to GPU if available
        with torch.no_grad():
            features = model(inputs).cpu()  # Extract features and move to CPU
        all_features.append(features)
        all_labels.append(labels)  # Save labels if needed for future reference

    # Concatenate all features and labels into a single tensor
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)  # Optional, if saving labels

    # Save as .pth file
    torch.save({'features': all_features, 'labels': all_labels}, filename)
    print(f"Features saved to {filename}")