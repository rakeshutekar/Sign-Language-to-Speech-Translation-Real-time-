import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to the training images
Alphabet_training_dir = './asl_alphabet_train'

# Initializing lists to hold average sizes and resolutions
avg_image_sizes = []
avg_image_resolutions = []
alphabets_lst = []

# Looping over each alphabet in the training directory
for alphabet in os.listdir(Alphabet_training_dir):
    print(f"Processing alphabet: {alphabet}")
    sizes = []
    resolutions = []
    
    # Looping over each image in the current category
    for image_name in os.listdir(os.path.join(Alphabet_training_dir, alphabet)):
        image_path = os.path.join(Alphabet_training_dir, alphabet, image_name)
        with Image.open(image_path) as img:
            width, height = img.size
            sizes.append(width * height)
            # Defaulting to 72 DPI if not present
            resolutions.append(img.info.get('dpi', (72, 72))[0])
    
    # Calculating the average size and resolution for the category
    avg_image_sizes.append(np.mean(sizes))
    avg_image_resolutions.append(np.mean(resolutions))
    alphabets_lst.append(alphabet)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(avg_image_sizes, avg_image_resolutions)
for i, txt in enumerate(alphabets_lst):
    plt.annotate(txt, (avg_image_sizes[i], avg_image_resolutions[i]))
plt.xlabel('Average Image Size (pixels)')
plt.ylabel('Average Image Resolution (DPI)')
plt.title('Average Image Size vs Resolution for Each alphabet')
plt.savefig('./Size_vs_Resolution.png', dpi=300, bbox_inches='tight')
plt.show()
