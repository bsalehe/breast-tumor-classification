import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BreastUltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transform

        for class_folder in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_folder)

            if not os.path.isdir(class_path):
                continue

            for fname in os.listdir(class_path):
                fpath = os.path.join(class_path, fname)
                if os.path.isfile(fpath):
                    if 'mask' in fname.lower():
                        self.mask_paths.append(fpath)
                    else:
                        self.image_paths.append(fpath)

        self.image_paths.sort()
        self.mask_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
