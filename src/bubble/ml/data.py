from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from bubble.domain import Concentration, FrameSample, ImageSize


class BubbleDataset(Dataset):
    def __init__(self, samples: list[FrameSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample.class_id


def get_data_transforms(image_size: ImageSize) -> dict[str, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    size_tuple = image_size.as_tuple()
    return {
        "train": transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize(size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(size_tuple),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }


def load_and_split_data(
    data_dir: str,
    random_state: int,
) -> tuple[list[FrameSample], list[FrameSample], list[FrameSample]]:
    train_samples: list[FrameSample] = []
    val_samples: list[FrameSample] = []
    test_samples: list[FrameSample] = []

    data_path = Path(data_dir)

    for concentration in Concentration:
        class_dir = data_path / concentration.label
        if class_dir.exists():
            img_paths = sorted(class_dir.glob("*.jpg"))
            n = len(img_paths)
            if n == 0:
                continue

            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            for img_path in img_paths[:train_end]:
                train_samples.append(FrameSample(path=img_path, concentration=concentration))
            for img_path in img_paths[train_end:val_end]:
                val_samples.append(FrameSample(path=img_path, concentration=concentration))
            for img_path in img_paths[val_end:]:
                test_samples.append(FrameSample(path=img_path, concentration=concentration))
        else:
            print(f"Warning: {class_dir} not found!")

    if not train_samples and not val_samples and not test_samples:
        raise ValueError(f"Error: No images found in {data_dir}")

    return train_samples, val_samples, test_samples