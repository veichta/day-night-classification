import cv2
import torch
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        """Initialize dataset.

        Args:
            data (list): list of image paths.
            labels (list): list of labels.
        """
        self.data = data

        self.images = []
        for i in tqdm(self.data):
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # resize image
            img = cv2.resize(img, (224, 224))

            if img is None:
                raise ValueError(f"Failed to load image: {i}")
            self.images.append(img)

        self.labels = labels

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __getitem__(self, index):
        # load image
        img = self.images[index]

        # convert to tensor
        img = torch.from_numpy(img)
        label = torch.tensor(self.labels[index])

        # normalize image
        img = img / 255.0
        img = (img - self.mean) / self.std

        img = img.permute(2, 0, 1)

        return img, label

    def __len__(self):
        return len(self.data)
