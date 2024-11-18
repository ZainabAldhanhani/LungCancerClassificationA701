# Comparative Analysis of Supervised Learning Algorithms for Lung Cancer Classification in CT Imaging
This repository contains the implementation and documentation

## Background
Lung cancer is one of the leading causes of cancer-related deaths worldwide, with an estimated
1.8 million deaths (18%). For improved patient care, early detection using imaging modalities
such as Computed Tomography (CT) is of utmost importance. CT imaging plays a vital role
in the identification of lung cancer, providing cross-sectional scans of the lungs that pronounce
the visualization of intricate pulmonary structures and abnormalities, such as nodules and masses.
However, the interpretation of CT images is often challenging due to the complexity and variability of
pathology presentations. Recent developments in the field of machine learning, specifically supervised
learning algorithms, demonstrate potential for automating and improving the categorization of lung
cancer from CT scans.
This project aims to establish a benchmarking classification result utilizing the chest CT-Scan images
dataset by evaluating the efficacy of various supervised learning algorithms in classifying lung cancer.
Through this analysis, we seek to identify the optimal models for clinical use, ensuring that the
chosen algorithms not only perform with high accuracy but also meet the practical requirements of
medical applications.
Our methodology begins by feeding the CT images into a selection of CNNs, which can include
pre-trained models for an interesting comparison, to extract the critical features that discern whether
the scans are indicative of cancer. Various classifiers will be explored for each CNN choice, including
Support Vector Machines (SVM) and Random Forests. Each classifier establishes a decision boundary
or imposes conditionals to differentiate between normal cases and the different types of cancer, hence
building the final model that will be tested for accurate lung cancer classification. Lastly, the
performance of these algorithms will be assessed based on metrics like accuracy, precision, and recall.
The results of this analysis will contribute to the growing body of literature on automated lung
cancer detection, providing insights that may facilitate the development of powerful diagnostic
tools. Ultimately, this research aspires to support clinicians in making informed decisions, thereby
improving early diagnosis and treatment strategies for lung cancer patients.
## Architecture
![Diagram](Figures/Diagram.png "Diagram")

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
## Implement The Full Pipeline  
```bash
cd 'Main Code'
python main.py 
```
## View Dataset Samples
```bash
cd visualization
python visualization.py --dataset_path Path
```
## Train and Evaluate All models
```bash
cd 'Train and evaluate all models'
python crossValidateTrain.py --train_dataset_path Path1 --test_dataset_path path2
```
## Test The Models on The Test Set And Save The Best Model
Model Weights are required in this step. Do the Train Step OR Download the weight that provided in this repository. 
```bash
cd 'Test the models on the test set'
python testDatasetonEachModel.py --train_dataset_path Path1 --test_dataset_path path2
```
> **Note**: Make sure that Model Weights in the same folder.

## Classification
For Random Forest: 
```bash
cd Classification
python RandomForest.py --train_dataset_path Path1 --test_dataset_path path2 --model_name model_name
```
For SVM: 
```bash
cd Classification
python SVM.py --train_dataset_path Path1 --test_dataset_path path2 --model_name model_name
```
> **Note**: Make sure that Model Weights in the same folder.
## Dataset Access

The dataset used for this project is available on Kaggle. It contains Chest CT scan images for  3 chest cancer types which are Adenocarcinoma,Large cell carcinoma, Squamous cell carcinoma , and 1 folder for the normal cell.
To access and download the dataset, follow these steps:

1. Visit the Kaggle dataset page: **[Chest CT Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)**
2. Click on the "Download" button to download the dataset manually, or use the Kaggle API to download it directly to your workspace.


## Model Weights

The model weights are hosted on OneDrive due to their large file size.  
You can download the weights by clicking the link below:

**[Access the Weights Here](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/zainab_aldhanhani_mbzuai_ac_ae/EtPDUCWLWddDkByjxYZjfxEBpC48W00Wf9uM7ZPSXlO7qw?e=3M92mh)**

> **Note**: Make sure you have adequate storage space and a stable internet connection when downloading the files.

## Methodology

## Contributions

## Challenges and Future Directions
