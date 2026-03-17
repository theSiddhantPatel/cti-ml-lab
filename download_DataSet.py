# To download the dataset to that exact folder, there is a slight catch we need to work around: the kagglehub.dataset_download() function doesn't actually have a parameter to set a custom destination folder yet. By default, it stubbornly downloads everything into a hidden cache folder on your computer.

# However, the easiest and cleanest way to get around this is to let kagglehub download it to the cache, and then use Python's built-in shutil library to instantly copy those files over to your specific C:\ path.

# Here is the complete script to do exactly that.

# The Python Script
# You can run this code in your IDE or save it as a .py script:

import kagglehub
import shutil
import os

# Your specific target destination (the 'r' prevents issues with Windows backslashes)
target_dir = r"C:\achievement\projects\cti-ml-lab\federated_cti\data"

# 1. Download the dataset (it downloads to a temporary cache folder first)
print("Downloading dataset...")
cache_dir = kagglehub.dataset_download("hassan06/nslkdd")

# 2. Automatically copy the contents of the cache to your specific folder
print(f"Moving files to {target_dir}...")
shutil.copytree(cache_dir, target_dir, dirs_exist_ok=True)

print(f"Success! Your data is ready to use at: {target_dir}")
