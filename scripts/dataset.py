
import os
from PIL import Image
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    """
    Custom Dataset for cancer cell images.
    Assumes directory structure:
        root_dir/
            benign/
                image1.jpg
                ...
            malignant/
                image101.jpg
                ...
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['benign', 'malignant']
        self.data = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.data.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label