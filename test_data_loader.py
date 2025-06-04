import matplotlib.pyplot as plt
from torchvision import transforms
from src.data_loader import BreastUltrasoundDataset

def show_sample(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image.permute(1, 2, 0))  # C x H x W → H x W x C
    axes[0].set_title('Ultrasound Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask.squeeze(), cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    plt.show()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = BreastUltrasoundDataset(root_dir='data/raw/Dataset_BUSI_with_GT', transform=transform)

    print(f"Dataset size: {len(dataset)}")

    # Show 3 random samples
    for i in range(3):
        image, mask = dataset[i]
        show_sample(image, mask)
