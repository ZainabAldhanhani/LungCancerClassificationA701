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

- `Classification/`: Classification Random Forest model and SVM Model
- `Dataset/`: This folder contains image examples for training dataset and testing dataset
- `Main Code/`: Main Code including all parts (Train and evaluate all models,Test the models on the test set,Visualization, and Classification )
- `Test the models on the test set/`: Test the models on the test set and save the best model
- `Train and evaluate all models`: Train and evaluate all models, get the best model and its name
- `Visualization/`: View dataset samples
- `README.md`: Overview and setup instructions.
- `requirements.txt`: Required libraries for the project.


## Install Requirements
Clone this repository and install the required Python packages:

```bash
git clone https://github.com/ZainabAldhanhani/LungCancerClassificationA701.git
cd LungCancerClassificationA701
pip install -r requirements.txt
```

## View Dataset Samples
```bash
cd visualization
python visualization.py --dataset_path Path
```
## Train and evaluate all models, get the best model and its name
```bash
cd 'Train and evaluate all models'
python crossValidateTrain.py --train_dataset_path Path1 --test_dataset_path path2
```
## Test the models on the test set and save the best model
Model Weights are required in this step. Do the Train Step OR Download the weight that provided in this repository. 
```bash
cd 'Test the models on the test set'
python testDatasetonEachModel.py --train_dataset_path Path1 --test_dataset_path path2
```
> **Note**: Make sure that Model Weights in the same folder.

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

