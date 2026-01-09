import os
from PIL import Image
from torch.utils.data import Dataset

class DAGMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        valid_ext = (".bmp", ".png", ".jpg")

        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(valid_ext) and "_label" not in f:
                    img_path = os.path.join(root, f)
                    base, ext = os.path.splitext(f)
                    label_file = base + "_label" + ext
                    label = 1 if label_file in files else 0
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
