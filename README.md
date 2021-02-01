# Digit Recognition

This repository includes 4 machine learning implementation to recognize digit. 

## Submitters: 

Yishay Seroussi 305027948, Samuel Bismuth 342533064.

## Python version:

 3.9

## Dataset

The dataset [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/overview) contains 40,000 samples of 28x28 images each of which represent a handwritten numerical digit. Note: the dataset used is the "train.csv" which was partitioned into a training subset and a testing subset.

## Machine Learning Techniques

- KNN
- Neural Network
- SVM
- K-Means Clustering

<!-- TODO CHANGE -->
## Configuration 

We use the docker environement. Make sure docker is installed in you machine. That is the only dependency of the project. 

According to your distribution, run:

    sudo <yum/apt-get> install -y docker

Then, to run all the techniques: 
    
    bash starts/start_all.sh 

To enter in the container terminal (only for developement purpose):

    bash starts/bash.sh 

Then, to run a specific technique: 
    
    bash starts/start_<technique>.sh 

If you don't want to use docker, you are able to run the code in any machine by folowing the next steps:

Install python3.9.

Install numpy by running:

    pip3 install repackage

Run the main python file of your choice:

    python3 main.py

## Code structure:

The code is composed of four folders.

- The data folder containing the csv file of data.
- The packages folder containing the requirement txt file with the pip lib we used.
- The starts folder containing all the starter scripts.
- The src folder containing the code source.
    - main.py -> The main file of the code. This is our entrypoint. This is also the file were the prints are done.

<!-- TODO -->
## Example of outputs:

## Work division

We worked on this code together using one computer as a pair programming.
That is, we've handle and understand together the complexity of the digit recognition mahcine learning techniques implementation and the code design in python. There is nothing in this work that have been done only by one submitter.
Notice that we worked only on one github account since we used only one computer.