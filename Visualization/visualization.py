
import matplotlib.pyplot as plt
import cv2
from dataset import resize_to_square
import argparse
# Visualization function for dataset
def visualize_dataset(df, n_samples=5,size=224):
# Define a color map for the labels to keep the visualization clear
    label_colors = {
        'normal': 'gray',
        'adenocarcinoma': 'red',
        'large.cell.carcinoma': 'blue',
        'squamous.cell.carcinoma': 'green'
    }

    fig, axes = plt.subplots(len(label_colors), n_samples, figsize=(15, 10))
    fig.suptitle("Dataset Samples by Category", fontsize=16)

    # Iterate through each label and plot n_samples images if available
    for i, (label, color) in enumerate(label_colors.items()):
        label_df = df[df['label'] == label]

        # Check if there are enough samples in the label category
        if len(label_df) >= n_samples:
            sample_images = label_df.sample(n=n_samples, random_state=42)
        elif len(label_df) > 0:
            sample_images = label_df  # If fewer samples, use all available
        else:
            print(f"Warning: No images found for category '{label}'. Skipping visualization for this category.")
            for j in range(n_samples):  # Empty plots for missing category
                axes[i, j].axis("off")
            continue

        for j, (_, row) in enumerate(sample_images.iterrows()):
            image = cv2.imread(row['path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = resize_to_square(image, size)  # Resize image for uniformity

            # Display image with label as title
            axes[i, j].imshow(image)
            axes[i, j].set_title(label, color=color)
            axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)
    plt.pause(1)

def main(dataset_path):
    dataset_df = get_profile_path(dataset_path)
    visualize_dataset(dataset_df)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Visualization  for dataset")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")
    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)