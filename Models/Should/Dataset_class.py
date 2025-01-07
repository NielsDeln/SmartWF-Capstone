
from torch.utils.data import Dataset, DataLoader

class Should_Dataset(Dataset):
    def __init__(self, data, labels, transforms=None) -> None:
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.labels[index]