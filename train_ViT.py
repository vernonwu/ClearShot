import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from ViT import DeblurTransformer
import torch.nn as nn
from PIL import Image as Image
import os
import torch
from basicsr.models.losses.losses import *


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, require_label=True):
        self.image_dir = image_dir
        self.datasets = os.listdir(os.path.join(image_dir, 'input'))
        self.image_list = []
        for dataset in self.datasets:
            image_list = os.listdir(os.path.join(image_dir, 'input', dataset))
            self.image_list += [os.path.join(dataset, x) for x in image_list]
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.require_label = require_label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        if self.require_label:
            label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))
        else:
            label = None

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = transforms.functional.to_tensor(image)
            if label is not None:
                label = transforms.functional.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            if not self.require_label:
                return image, name
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

# Assume you have lists of blurred and sharp images
blurred_images = [...]  # List of blurred images
sharp_images = [...]  # List of sharp images

# Transform to convert images to tensors
transform = transforms.Compose([ToTensor()])
data_dir = "./media/val/"

# Create dataset and dataloader
dataset = DeblurDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = DeblurTransformer().to(torch.device("cuda"))
criterion1 = L1Loss()
criterion2 = FFTLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for blurred, sharp in dataloader:
        optimizer.zero_grad()
        blurred = blurred.to(torch.device("cuda"))
        sharp = sharp.to(torch.device("cuda"))
        output = model(blurred)
        loss1 = criterion1(output, sharp)
        loss2 = criterion2(output, sharp)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

torch.save(model.state_dict(), "./results/ViT.pt")
