import pandas as pd
from pathlib import Path
import os

filepath = 'labels.csv'
new_directory = '/home/VLM-backdoor/DeDe-main/src/BadCLIP/data/ImageNet1K/validation/ILSVRC2012_img_val'

labels = pd.read_csv(filepath)

for row in range(len(labels)):
    imagepath = labels.loc[row, 'image']
    imagename = Path(imagepath).name
    newimagepath = os.path.join(new_directory, imagename)
    labels.loc[row, 'image'] = newimagepath

labels.to_csv(filepath, index=False)