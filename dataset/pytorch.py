import json
import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset

class PigeonBBoxDataset(Dataset):
    def __init__(self, json_path: str, images_root: str, transform=None, target_transform=None, normalize_bboxes: bool = False):
        """
        Args:
            json_path: Path to the VIA JSON file.
            images_root: Directory where the image files (filenames from JSON) are stored.
            transform: Optional torchvision transform for the image.
            target_transform: Optional transform for the bounding box tensor.
            normalize_bboxes: If True, returns bbox normalized to [0,1]
                              relative to image width/height: [x/W, y/H, w/W, h/H].
        """
        self.images_root = images_root
        self.transform = transform
        self.target_transform = target_transform
        self.normalize_bboxes = normalize_bboxes
        self.negative_bbox = [0, 0, 0, 0, 0]

        # Load the full JSON annotation dict
        with open(json_path, "r") as f:
            annotations = json.load(f)

        self.samples = []

        for _, entry in annotations.items():
            filename = entry.get("filename")
            regions = entry.get("regions", [])

            img_path = os.path.join(images_root, filename)
            if not os.path.isfile(img_path):
                continue

            # Collect ALL valid regions for this image
            bboxes = []
            for region in regions:
                shape = region.get("shape_attributes", {})
                x = shape.get("x")
                y = shape.get("y")
                w = shape.get("width")
                h = shape.get("height")

                if None in (x, y, w, h):
                    continue

                bboxes.append([1, x, y, w, h])

            # Use images with no valid regions as negative examples
            if not bboxes:
                bboxes.append(self.negative_bbox)

            self.samples.append(
                {
                    "img_path": img_path,
                    "bboxes": bboxes,  # list of [confidence, x, y, w, h]
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        bboxes = sample["bboxes"]  # list of bboxes for this image

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Pick ONE region at random for this sample
        bbox = random.choice(bboxes)  # [confidence, x, y, w, h] in pixels (original image)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Convert bbox to tensor [confidence, x, y, w, h]
        bbox = torch.tensor(bbox, dtype=torch.float32)

        if self.normalize_bboxes:
            prob, x, y, w, h = bbox
            bbox = torch.tensor(
                [
                    prob,
                    x / orig_width,
                    y / orig_height,
                    w / orig_width,
                    h / orig_height,
                ],
                dtype=torch.float32,
            )

        if self.target_transform is not None:
            bbox = self.target_transform(bbox)

        return image, bbox
