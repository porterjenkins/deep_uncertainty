from torch.utils.data import Dataset


class ImageDatasetWrapper(Dataset):
    """A wrapper class that allows for assigning different transforms to different views of the same dataset (such as train/test)."""

    def __init__(self, base_dataset, transforms):
        super(ImageDatasetWrapper, self).__init__()
        self.base = base_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transforms(x), y
