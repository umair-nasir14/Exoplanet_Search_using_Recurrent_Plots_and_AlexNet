# Exoplanet_Search_using_Recurrent_Plots_and_AlexNet

**********CODE READABILITY GUIDE**********


-> main.py is the main file to be executed to run the code.
-> alexnet.py has the AlexNet model and the utilities for the model, runs through main.py.
-> dataprocessor.py processes the data, runs through main.py
-> extractrecuplots.py extracts recurrence plots.
-> extracttimeseriesgraph extracts light curves.
-> dataaug.py augments the images.
-> undersampling.py deletes randome images.

Main requirements are Tensorflow 2.x and PyTS

Model was trained on:

-> GPU: GTX 1050
-> CPU: AMD RYZEN 5 and Intel Core i7 (both 4 cores)
-> RAM: 16GB

