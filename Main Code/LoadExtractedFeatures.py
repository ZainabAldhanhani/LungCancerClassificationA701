# Load the extracted features and labels for the training set
import torch
import torchvision
from loadImagepaths import *
from extractSaveFeatures import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_extractedfeatures_train_and_test(dataset_df,test_df, model_name):

    model_weights = {
    'vgg16': 'VGG16_Weights.DEFAULT',
    'resnet50': 'ResNet50_Weights.DEFAULT',
    'densenet121': 'DenseNet121_Weights.DEFAULT'
}
    
    model = getattr(torchvision.models, model_name)(weights=model_weights[model_name]).to(device)
    model.load_state_dict(torch.load(f'{model_name}_weights.pth',map_location=torch.device("cuda" if torch.cuda.is_available() else 'cpu')), strict=False)
    model.eval()
    
    extract_and_save_features(dataset_df, 'extract_train.pth', model)
    train_data = torch.load('extract_train.pth')
    X_train = train_data['features'].numpy()  # Convert to numpy array for sklearn
    y_train = train_data['labels'].numpy()

    extract_and_save_features(test_df, 'extract_test.pth', model)
    test_data = torch.load('extract_test.pth')
    X_test = test_data['features'].numpy()  # Convert to numpy array for sklearn
    y_test = test_data['labels'].numpy()
    return X_train, y_train, X_test, y_test