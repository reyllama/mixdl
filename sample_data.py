import shutil, os
from tqdm import tqdm
import random

# source_name = "AnimalFace-dog"
# target_name = "Dog"

# source_path = f"/ssd2/research/angels/Dataset/full/dog/img/"
source_path = "/ssd2/research/angels/Dataset/full/oxford-flowers/img/"
target_path  = "/ssd2/research/angels/Dataset/selected/flower-1000-shot/img/"

n_shots = 1000

files = os.listdir(source_path)

chosen_files = random.sample(files, k=n_shots)

if not os.path.isdir(target_path):
    os.makedirs(target_path)

for f in tqdm(chosen_files):
    shutil.copyfile(os.path.join(source_path, f), os.path.join(target_path, f))