import os
import shutil

IMG_DIR = "../dataset/raw"
TEST_DIR = "../dataset/tests"

os.makedirs(TEST_DIR,exist_ok=True)

img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith('jpg')]

selected_files = img_files[:2]

for img_file in selected_files:
    img_path = os.path.join(IMG_DIR, img_file)
    test_path = os.path.join(TEST_DIR, img_file)

    shutil.copy2(img_path, test_path)

