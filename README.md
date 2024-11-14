# Comparative Analysis of Supervised Learning Algorithms for Lung Cancer Classification in CT Imaging

**Authors**  
- **Aamna Alshehhi**  
  Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)  
  [aamna.alshehhi@mbzuai.ac.ae](mailto:aamna.alshehhi@mbzuai.ac.ae)

- **Fatema Alkamali**  
  Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)  
  [fatema.alkamali@mbzuai.ac.ae](mailto:fatema.alkamali@mbzuai.ac.ae)

- **Zainab Aldhanhani**  
  Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)  
  [zainab.aldhanhani@mbzuai.ac.ae](mailto:zainab.aldhanhani@mbzuai.ac.ae)

---
## Repository Structure

- `Data Pre and Post Processing/`: Source code containing all scripts for pre-processing the histopathology images including stain normalization and the mask post-processing
- `Classification/`: This folder contains code to train and test the baseline classification models including ResNet50, DenseNet121, and ViT
- `Segmentation/`: Source code for training the DeepLabV3 segmentation model and inferencing the segmentation masks
- `Radiomics Features/`: Source code to extract the radiomics features of white blood cells using generated masks
- `Multi-Modal Model/`: This folder contains code to train the multi-modal model with ViT and ConvNext model along with radiomics tabular features
- `Mask Attention/`: Source code for ablation experiment to use mask attention to aid in classification
- `Tabular Attention/`: Source code for ablation experiment to implement tabular attention
- `Multiple Instance Learning/`: This folder contains code to train the MIL model to patchify histopathology images and use bagging to predict diagnosis
- `data/`: This folder contains demo input and output files to test the code
- `README.md`: Overview and setup instructions.
- `requirements.txt`: Required libraries for the project.


## Install Requirements

To install all necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```
## Dataset Access

The dataset used for this project is available on Kaggle. It contains Chest CT scan images for various medical imaging tasks.

To access and download the dataset, follow these steps:

1. Visit the Kaggle dataset page: **[Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)**
2. Click on the "Download" button to download the dataset manually, or use the Kaggle API to download it directly to your workspace.


## Model Weights

The model weights are hosted on OneDrive due to their large file size.  
You can download the weights by clicking the link below:

**[Access the Weights Here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zainab_aldhanhani_mbzuai_ac_ae/EtPDUCWLWddDkByjxYZjfxEBpC48W00Wf9uM7ZPSXlO7qw?e=3M92mh)**

> **Note**: Make sure you have adequate storage space and a stable internet connection when downloading the files.

