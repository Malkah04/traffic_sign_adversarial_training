from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR 
TEST_CSV = BASE_DIR / "Test.csv"
IMAGE_SIZE = 32
BATCH_SIZE = 128
NUM_CLASSES = 43
VAL_SPLIT = 0.2
SEED = 42

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

class GTSRBTestDataset(Dataset):

    def __init__(self, test_dir, csv_file, transform=None):

        self.test_dir = Path(test_dir)
        self.transform = transform
        self.data = pd.read_csv(csv_file)

        print("\nTest.csv columns:")
        print(self.data.columns.tolist())
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # print("self.test_dir" ,self.test_dir )
        # print("else", self.test_dir / row["Path"])
        image_path = self.test_dir / row["Path"]

        image = Image.open(image_path).convert("RGB")

        label = int(row["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders():

    full_train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform ,
        
    )
    class_to_actual_id = [int(c) for c in full_train_dataset.classes]
    
    # Apply target transform using pre-built class ID mapping
    full_train_dataset.target_transform = lambda y: class_to_actual_id[y]

    test_dataset = GTSRBTestDataset(
        test_dir=TEST_DIR,
        csv_file=TEST_CSV,
        transform=test_transform
    )
    print(test_dataset[0])
  

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


train_loader, test_loader = get_dataloaders()

print("TRAFFIC SIGN DATASET")

print(f"Train samples : {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

print(f"Number of classes: {NUM_CLASSES}")


images, labels = next(iter(train_loader))

print("\nBatch information:")
print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")

print("\nPixel range:")
print(f"Min : {images.min().item():.4f}")
print(f"Max : {images.max().item():.4f}")
print(images[0])
print(labels[0])
print()

images, labels = next(iter(test_loader))


print("test" ,images[0])
print("test", labels[0])