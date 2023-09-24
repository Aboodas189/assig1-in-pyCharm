import os
from random import *
import nnfs
import numpy as np
import matplotlib.pyplot as plt
import math
from nnfs.datasets import spiral_data
from Layer import Layer
from os import listdir
from os.path import isfile, join


def A1(numberOfFiles):
    path = "SmallDataset"
    y = []
    X = []
    # Get the list of folders
    folders = os.listdir(path)
    for folder in folders:
        # get the list of files
        files = os.listdir(path + "//" + folder)
        i = 0
        for file in files:
            # read the current file
            img = plt.imread(path + "//" + folder + "//" + file)
            # Change from an image to an array
            arr = np.array(img, dtype=np.float64)
            # flatten the array into gray scale
            flat_arr = arr.ravel()
            # append into the ground truth list, but you need to cast it into an int first
            y.append(int(folder))
            # append into the feature matrix as a list
            X.append(flat_arr.tolist())
            i += 1
            if (i == numberOfFiles):
                break
    # swap both to np array because that is what the layer works on
    y = np.array(y)
    X = np.array(X)
    return X, y, len(folders), X.shape[1]


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def StartNN():
    # Create dataset
    # Vertical data doesn't work, looked it up online as well, no docuemntation for it
    # X, y = spiral_data(samples=100, classes=3)
    # X, y= vertical_data(samples=100, classes=3)
    numberOfFiles = 2000
    numberOfFiles = clamp(numberOfFiles, 500, 999) # from 500 to 999
    X, y, nClasses, nFeatures = A1(numberOfFiles)
    # neurons1 = randint(50, 100)
    # neurons2 = randint(10, 49)
    neurons1 = 50
    neurons2 = 25
    numIteration = 100
    inputLayer = Layer(nFeatures, neurons1)
    hiddenLayer1 = Layer(neurons1, neurons2)
    outputLayer = Layer(neurons2, nClasses)
    # inputLayer = Layer(2, 4)
    # outputLayer = Layer(4, 3)
    lowest_loss = 9999999  # some initial value
    lowest_accuracy = -9999999  # some initial value
    training = 0.005
    best_input_weights = inputLayer.weights.copy()
    best_input_biases = inputLayer.biases.copy()
    best_hidden1_weights = hiddenLayer1.weights.copy()
    best_hidden1_biases = hiddenLayer1.biases.copy()
    best_output_weights = outputLayer.weights.copy()
    best_output_biases = outputLayer.biases.copy()
    i = 0
    for iteration in range(numIteration):
        # Update weights with some small random values
        inputLayer.weights += training * np.random.randn(nFeatures, neurons1)
        inputLayer.biases += training * np.random.randn(1, neurons1)
        hiddenLayer1.weights += training * np.random.randn(neurons1, neurons2)
        hiddenLayer1.biases += training * np.random.randn(1, neurons2)
        outputLayer.weights += training * np.random.randn(neurons2, nClasses)
        outputLayer.biases += training * np.random.randn(1, nClasses)
        inputLayer.forward(X)
        inputLayer.ApplyReLU()
        hiddenLayer1.forward(inputLayer.output)
        hiddenLayer1.ApplyReLU()
        outputLayer.forward(hiddenLayer1.output)
        outputLayer.ApplySoftMax()
        loss = outputLayer.CalulateLoss(y)
        accuracy = outputLayer.CalulateAccuracy(y)
        # accuracy = 0
        if loss < lowest_loss:
            # if lowest_accuracy < accuracy:
            print('New set of weights found, iteration:', iteration,
                  'loss:', loss, 'acc:', accuracy)
            i += 1;
            best_input_weights = inputLayer.weights.copy()
            best_input_biases = inputLayer.biases.copy()
            best_hidden1_weights = hiddenLayer1.weights.copy()
            best_hidden1_biases = hiddenLayer1.biases.copy()
            best_output_weights = outputLayer.weights.copy()
            best_output_biases = outputLayer.biases.copy()
            lowest_loss = loss
            lowest_accuracy = accuracy
        else:
            inputLayer.weights = best_input_weights.copy()
            inputLayer.biases = best_input_biases.copy()
            hiddenLayer1.weights = best_hidden1_weights.copy()
            hiddenLayer1.biases = best_hidden1_biases.copy()
            outputLayer.weights = best_output_weights.copy()
            outputLayer.biases = best_output_biases.copy()
    print("DONE!", str(neurons1), str(neurons2), str(i))


if __name__ == "__main__":
    # A1()
    StartNN()
