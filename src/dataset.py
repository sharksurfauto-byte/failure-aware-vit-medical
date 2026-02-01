"""
PyTorch Dataset for NIH Malaria with locked preprocessing pipeline.

Preprocessing:
1. Load image (RGB)
2. Aspect-ratio preserving resize (shorter side → 224px)
3. Symmetric black padding to 224×224
4. Convert to tensor
5. Normalize with dataset-specific mean/std (computed on train split only)
"""

import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class MalariaDataset(Dataset):
    """NIH Malaria Dataset with locked preprocessing pipeline"""
    
    # Dataset-specific normalization stats (computed from train split)
    # Will be updated after first computation
    MEAN = None
    STD = None
    
    def __init__(self, split='train', splits_path='data/splits.json', compute_stats=False):
        """
        Args:
            split: 'train', 'val', 'test_clean', or 'stress_source'
            splits_path: Path to splits.json file
            compute_stats: If True, compute dataset stats (only for train split)
        """
        assert split in ['train', 'val', 'test_clean', 'stress_source'], \
            f"Invalid split: {split}"
        
        self.split = split
        self.compute_stats = compute_stats
        
        # Load splits
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)
        
        # Get image paths for this split
        if split == 'test_clean':
            self.image_paths = splits_data['splits']['test_clean']
        elif split == 'stress_source':
            self.image_paths = splits_data['splits']['stress_source']
        else:
            self.image_paths = splits_data['splits'][split]
        
        # Get small images metadata
        self.small_images = set(splits_data['metadata']['small_images'])
        
        # Define transforms
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Setup preprocessing transforms"""
        # Resize + pad transform
        self.resize_pad = ResizePad(target_size=224, pad_value=0)
        
        # To tensor
        self.to_tensor = transforms.ToTensor()
        
        # Normalization (will be set later if computing stats)
        if self.compute_stats:
            # Don't normalize when computing stats
            self.normalize = None
        else:
            # Use precomputed stats (must be set before using dataset)
            if MalariaDataset.MEAN is None or MalariaDataset.STD is None:
                raise ValueError(
                    "Dataset normalization stats not set. "
                    "Please compute stats first using compute_normalization_stats()"
                )
            self.normalize = transforms.Normalize(
                mean=MalariaDataset.MEAN,
                std=MalariaDataset.STD
            )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """Get image and label"""
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get label from path (Parasitized=1, Uninfected=0)
        label = 1 if 'Parasitized' in img_path else 0
        
        # Check if small image
        is_small = img_path in self.small_images
        
        # Apply preprocessing
        image = self.resize_pad(image)
        image = self.to_tensor(image)
        
        if self.normalize is not None:
            image = self.normalize(image)
        
        # Return image, label, metadata
        return image, label, {'path': img_path, 'is_small': is_small}
    
    @staticmethod
    def compute_normalization_stats(train_dataset, num_samples=5000):
        """
        Compute mean and std from train split.
        
        Args:
            train_dataset: MalariaDataset instance for train split (with compute_stats=True)
            num_samples: Number of random samples to use for estimation
        
        Returns:
            mean, std (both as lists)
        """
        print(f"Computing normalization stats from {num_samples} train samples...")
        
        # Use subset for efficiency
        indices = np.random.choice(len(train_dataset), 
                                   min(num_samples, len(train_dataset)), 
                                   replace=False)
        
        pixel_values = []
        
        for idx in indices:
            img, _, _ = train_dataset[idx]
            pixel_values.append(img.numpy())
        
        # Stack and compute stats
        pixel_values = np.stack(pixel_values)  # Shape: (N, C, H, W)
        
        mean = pixel_values.mean(axis=(0, 2, 3))  # Mean per channel
        std = pixel_values.std(axis=(0, 2, 3))    # Std per channel
        
        print(f"Computed mean: {mean}")
        print(f"Computed std: {std}")
        
        # Update class variables
        MalariaDataset.MEAN = mean.tolist()
        MalariaDataset.STD = std.tolist()
        
        return mean, std


class ResizePad:
    """Aspect-ratio preserving resize + symmetric black padding"""
    
    def __init__(self, target_size=224, pad_value=0):
        """
        Args:
            target_size: Target square size (default 224×224)
            pad_value: Padding value (default 0 = black)
        """
        self.target_size = target_size
        self.pad_value = pad_value
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        
        Returns:
            PIL Image of size (target_size, target_size)
        """
        w, h = img.size
        
        # Compute scale factor (resize shorter side to target_size)
        scale = self.target_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize with bilinear interpolation
        img = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Compute padding
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        
        # Symmetric padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Apply padding
        padding = (pad_left, pad_top, pad_right, pad_bottom)
        img = transforms.functional.pad(img, padding, fill=self.pad_value)
        
        return img


# Initialize normalization stats from train split (will be updated on first use)
def initialize_normalization_stats(splits_path='data/splits.json', save_path='data/normalization_stats.json'):
    """Compute and save normalization stats"""
    # Create temporary dataset for stats computation
    train_dataset = MalariaDataset(split='train', splits_path=splits_path, compute_stats=True)
    
    # Compute stats
    mean, std = MalariaDataset.compute_normalization_stats(train_dataset, num_samples=5000)
    
    # Save to file
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(save_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[OK] Saved normalization stats to {save_path}")
    return mean, std


def load_normalization_stats(stats_path='data/normalization_stats.json'):
    """Load precomputed normalization stats"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    MalariaDataset.MEAN = stats['mean']
    MalariaDataset.STD = stats['std']
    
    print(f"Loaded normalization stats: mean={stats['mean']}, std={stats['std']}")


if __name__ == "__main__":
    # Initialize normalization stats
    print("Initializing dataset normalization stats...")
    initialize_normalization_stats()
    
    # Test dataset
    print("\nTesting dataset loading...")
    load_normalization_stats()
    
    train_ds = MalariaDataset('train')
    val_ds = MalariaDataset('val')
    test_ds = MalariaDataset('test_clean')
    
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")
    
    # Test single sample
    img, label, meta = train_ds[0]
    print(f"\nSample shape: {img.shape}")
    print(f"Label: {label}")
    print(f"Is small: {meta['is_small']}")
