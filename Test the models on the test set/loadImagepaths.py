import pandas as pd
import os
# Define function to load image paths and labels from dataset folder structure

def get_profile_path(root_dir):
    data = []
    classes = ['normal', 'adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                for category in classes:
                    if category in root:
                        data.append({
                            'ImageID': file,
                            'path': os.path.join(root, file),
                            'label': category
                        })
                        break
    return pd.DataFrame(data)

