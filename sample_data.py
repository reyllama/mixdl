import shutil, os

source_name = "AnimalFace-dog"
target_name = "Dog"

# source_path = f"/ssd2/research/angels/Dataset/full/dog/img/"
source_path = "/ssd2/research/angels/Dataset/full/baby_images/"
target_path  = "/ssd2/research/angels/Dataset/selected/baby-30-shot/img/"


import random

n_shots = 30

files = os.listdir(source_path)

chosen_files = random.sample(files, k=n_shots)

for f in chosen_files:
    shutil.copyfile(source_path+f, target_path+f)


