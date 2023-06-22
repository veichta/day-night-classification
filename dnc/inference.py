from typing import Dict

import cv2
import numpy as np
import torch

from dnc.model import Classifier

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


def load_image(img_path: str) -> np.ndarray:
    """Load image.

    Args:
        img_path (str): path to image.

    Returns:
        img (np.ndarray): image as numpy array.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    return img


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Load image as tensor.

    Args:
        img_path (str): path to image.

    Returns:
        img (torch.Tensor): image as tensor."""
    img = cv2.resize(img, (224, 224))

    # convert to tensor
    img = torch.from_numpy(img)

    # normalize image
    img = img / 255.0
    img = (img - MEAN) / STD

    # permute from (H, W, C) to (C, H, W)
    img = img.permute(2, 0, 1)

    return img


def infer(
    img: np.ndarray,
    backbone: str = "resnet18",
    weights: str = "models/resnet18.pt",
    device: str = "cpu",
) -> Dict[str, float]:
    """Predict day or night from image.

    Args:
        img (np.ndarray): image as numpy array.
        backbone (str, optional): backbone architecture. Defaults to "resnet18".
        weights (str, optional): path to weights. Defaults to "models/resnet18.pt".
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        Dict[str, float]: prediction.
    """
    img = image_to_tensor(img)
    model = Classifier(backbone=backbone)
    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

    model.eval()
    model.to(device)
    with torch.no_grad():
        pred = model(img.unsqueeze(0))
        pred = torch.softmax(pred, dim=1)
        pred = pred.squeeze(0).cpu().numpy()

    return {"day": pred[1], "night": pred[0]}


def infer_from_file(
    img_path: str,
    backbone: str = "resnet18",
    weights: str = "models/resnet18.pt",
    device: str = "cpu",
) -> Dict[str, float]:
    """Predict day or night.

    Args:
        img_path (str): path to image.
        backbone (str, optional): backbone architecture. Defaults to "resnet18".
        weights (str, optional): path to weights. Defaults to "models/resnet18.pt".
        device (str, optional): device to use. Defaults to "cpu".

    Returns:
        Dict[str, float]: prediction.
    """
    img = load_image(img_path)
    return infer(img, backbone=backbone, weights=weights, device=device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--weights", type=str, default="models/resnet18.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    pred = infer_from_file(
        img_path=args.img_path,
        backbone=args.backbone,
        weights=args.weights,
        device=args.device,
    )
    print(pred)
