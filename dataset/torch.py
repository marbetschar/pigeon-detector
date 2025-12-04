import json
import os
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

        # Load the full JSON annotation dict
        with open(json_path, "r") as f:
            annotations = json.load(f)

        self.samples = []

        # Each entry looks like:
        # "2025-07-08_07-39-49.jpg110533": {
        #   "filename": "2025-07-08_07-39-49.jpg",
        #   "regions": [ { "shape_attributes": { "x": ..., "y": ..., "width": ..., "height": ... } } ]
        # }
        na_bbox = [0, 0, 0, 0, 0]

        for _, entry in annotations.items():
            filename = entry.get("filename")
            regions = entry.get("regions", [])

            bbox = na_bbox

            # Skip if no regions
            if regions:
                # Use the first region only
                shape = regions[0].get("shape_attributes", {})
                x = shape.get("x")
                y = shape.get("y")
                w = shape.get("width")
                h = shape.get("height")

                # Skip if any coordinate is missing
                if None not in (x, y, w, h):
                    bbox = [1, x, y, w, h]

            img_path = os.path.join(images_root, filename)

            # Optionally skip if the file does not actually exist
            if not os.path.isfile(img_path):
                continue

            self.samples.append(
                {
                    "img_path": img_path,
                    "bbox": bbox
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["img_path"]
        bbox = sample["bbox"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Convert bbox to tensor [x, y, w, h]
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
