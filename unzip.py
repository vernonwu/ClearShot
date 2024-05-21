import zipfile
import os
input_dir = './REDS/val_sharp.zip'
output_dir = './media'
with zipfile.ZipFile(input_dir, 'r') as zip_ref:
    zip_ref.extractall(output_dir)