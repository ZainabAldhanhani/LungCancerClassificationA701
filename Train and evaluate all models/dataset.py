import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
# Dataset class for loading images

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, size):
        self.df = df
        self.size = size
        self.labels_map = {label: idx for idx, label in enumerate(['normal', 'adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma'])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_to_square(image, self.size)
        image = pad(image, self.size, self.size)
        tensor = image_to_tensor(image, normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
        label = self.labels_map[row.label]
        return tensor, label


# Resize function for images
def resize_to_square(image, size):
    h, w, d = image.shape
    ratio = size / max(h, w)
    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)
    return resized_image

# Convert image to tensor
def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

# Pad image to specific dimensions
def pad(image, min_height, min_width):
    h, w, d = image.shape
    h_pad_top = max((min_height - h) // 2, 0)
    h_pad_bottom = min_height - h - h_pad_top
    w_pad_left = max((min_width - w) // 2, 0)
    w_pad_right = min_width - w - w_pad_left
    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
