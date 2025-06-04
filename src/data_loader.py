import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class BreastUltrasoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all images and masks, e.g. 'data/raw/Dataset_BUSI_with_GT'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Assume images in 'data' subfolder and masks in 'mask' subfolder
        #self.image_dir = os.path.join(root_dir, 'data')
        self.image_dir = root_dir
        #self.mask_dir = os.path.join(root_dir, 'mask')
        self.mask_dir = root_dir  # same as image_dir

        self.image_names = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)  # mask has same name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
