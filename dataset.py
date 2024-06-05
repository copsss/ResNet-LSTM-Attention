import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class InjectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['injSuccess', 'injFail']
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for seq_name in os.listdir(class_dir):
                seq_dir = os.path.join(class_dir, seq_name)
                images = [os.path.join(seq_dir, img) for img in sorted(os.listdir(seq_dir))]
                self.data.append((images, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, label = self.data[idx]
        imgs = [Image.open(img).convert('RGB') for img in images]

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        imgs = torch.stack(imgs)
        return imgs, label
