import zipfile
import os
input_dir = './REDS/val_blur.zip'
output_dir = './media/val/test'
with zipfile.ZipFile(input_dir, 'r') as zip_ref:
    zip_ref.extractall(output_dir)