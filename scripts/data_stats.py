from pathlib import Path

data_root = Path("data/raw/cell_images")

parasitized = list((data_root / "Parasitized").glob("*.png"))
uninfected = list((data_root / "Uninfected").glob("*.png"))

print(f"Parasitized: {len(parasitized)}")
print(f"Uninfected: {len(uninfected)}")
print(f"Total: {len(parasitized) + len(uninfected)}")

import cv2
import numpy as np
from tqdm import tqdm

shapes = []

for img_path in parasitized + uninfected:
    img = cv2.imread(str(img_path))
    h, w, c = img.shape
    shapes.append((h, w))

shapes = np.array(shapes)

print("Min H, W:", shapes.min(axis=0))
print("Max H, W:", shapes.max(axis=0))
print("Mean H, W:", shapes.mean(axis=0))

pixels = []

for img_path in parasitized[:500] + uninfected[:500]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    pixels.append(img.flatten())

pixels = np.concatenate(pixels)

print("Pixel stats:")
print("Min:", pixels.min())
print("Max:", pixels.max())
print("Mean:", pixels.mean())
print("Std:", pixels.std())

small=[]
for cls in ['Parasitized', 'Uninfected']:
    for p in (data_root/ cls).glob("*.png"):
        img = cv2.imread(str(p))
        h,w,_ = img.shape
        if h < 80 or w < 80:
            small.append((p,h,w))

print(f"Images < 80px: {len(small)}")
print(f"Percentage: {len(small)/27558*100:.2f}%")

# Print a few examples
for s in small[:5]:
    print(s)
