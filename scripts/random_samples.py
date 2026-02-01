import os
import random
import matplotlib.pyplot as plt

parasitized_dir = 'data/raw/cell_images/Parasitized'
uninfected_dir = 'data/raw/cell_images/Uninfected'

# randomly sampling 20 parasitized images:
parasitized_samples = random.sample(os.listdir(parasitized_dir), 20)

for sample in parasitized_samples:
    img = plt.imread(os.path.join(parasitized_dir, sample))
    plt.imshow(img)
    plt.show()

# randomly sampling 20 uninfected images:
uninfected_samples = random.sample(os.listdir(uninfected_dir), 20)

for sample in uninfected_samples:
    img = plt.imread(os.path.join(uninfected_dir, sample))
    plt.imshow(img)
    plt.show()