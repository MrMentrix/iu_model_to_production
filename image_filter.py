from PIL import Image
import os
from tqdm import tqdm

base_dir = os.path.basename("training_data_small")
image_dir = os.path.join(base_dir, "images")

deleted = []

for i, image in tqdm(enumerate(os.listdir(image_dir))):
    if i == 0:
        continue
    # if i == 0:
    #     break

    # This makes sure that all images have 3 color channels
    img = Image.open(os.path.join(image_dir, image))
    if img.mode != "RGB":
        # delete image
        deleted.append(image)

for image in deleted:
    os.remove(os.path.join(image_dir, image))

print(deleted)
print(f"Deleted {len(deleted)} images.")