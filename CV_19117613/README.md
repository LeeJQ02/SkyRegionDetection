# This is for Computer Vision Project CSC 3014 (Sky Region Detection Algorithm)
This is a Python script that implements a sky region deetction algorithm using 
computer vision techniques. The algorithm is designed to detect the sky region
in outdoor images and include steps for operation.

## Requirement
- Make sure python is install on your device. 
- Python version can be 3.9 or later. 
- You can check from cmd (Windows) or terminal (Mac) by typing 'python --version'
command or simply have a python IDE, Spyder IDE or any IDE support python.
- Make sure you have download datasets 623, 684, 9730 and 10917 and their mask from [Skyfinder Cameras](https://cs.valdosta.edu/~rpmihail/skyfinder/images/index.html)
- After downloading, place the all the dataset image folder into a folder called datasets and a mask file into a folder called mask folder.

## How to run and execute program
1. Make sure the script "CV2_19117613.py" is in the same directory as the downloaded images and mask.
2. Open the script "CV2_19117613.py" from your chosen IDE and simply click run and the program will be executed and output will be saved. 
    1. All the figures will be downloaded into a folder called results.
    2. For only displaying figures, from line 124 to line 130 can be commented to avoid from saving into the results folder. 
    3. For only outputing the accuracies in the console, from line 110 to line 131 can be commented to only calculate the accuracy

