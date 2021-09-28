import os
import random

'''

   This file undersamples the Non panet observations to make our data balanced.

'''


path = r'C:/Semester-1/ACML/Project-ExoPlanet-Search/Latest/exoPlanet_data_1.1/Aug_data/Non-planet'    

#for folder in folder_paths:  Go over each folder path
files = os.listdir(path)  # Get filenames in current folder
files = random.sample(files, 2500)  # Pick 2500 random files
for file in files:  # Go over each file name to be deleted
    f = os.path.join(path, file)  # Create valid path to file
    os.remove(f)  # Remove the file